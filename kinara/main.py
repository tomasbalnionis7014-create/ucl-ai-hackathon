import os
import cv2
import time
import tempfile
import shutil
import json
import dataclasses
import requests
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your actual classes
from pose_detector import PoseDetector
from metrics_analyzer import MetricsAnalyzer
from video_annotator import VideoAnnotator

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- CLIENTS ---
local_client = OpenAI(base_url="http://localhost:8000/v1",
                      api_key="token-not-needed")
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# Ensure output directory exists for saved videos
os.makedirs("outputs", exist_ok=True)

# Define the expected JSON payload from frontend


class VideoRequest(BaseModel):
    videoUrl: str
    generateVideo: bool = False  # Added toggle


def sample_turbo_frames(input_path: str, output_path: str, target_frames: int = 8):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    scale = 224 / orig_h
    new_w, new_h = int(orig_w * scale), 224

    step = max(1, total_frames // target_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))

    count = 0
    for i in range(0, total_frames, step):
        if count >= target_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_NEAREST)
        out.write(resized)
        count += 1

    cap.release()
    out.release()


@app.post("/full-analysis")
async def process_full_pipeline(req: VideoRequest):
    print(f"\n📥 [START] Pulling video from Cloudinary: {req.videoUrl}")
    overall_start = time.time()

    # Download the video from Cloudinary directly to Brev
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir="/tmp") as temp_in:
        try:
            response = requests.get(req.videoUrl, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                temp_in.write(chunk)
            in_path = temp_in.name
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to fetch video: {str(e)}")

    out_path = in_path.replace(".mp4", "_turbo.mp4")

    # Define a permanent save path on the Brev server using the URL's filename
    safe_filename = req.videoUrl.split("/")[-1].split("?")[0]
    if not safe_filename.endswith(".mp4"):
        safe_filename += ".mp4"
    annotated_path = f"outputs/annotated_{int(time.time())}_{safe_filename}"

    try:
        # STEP 1: LOCAL VISION
        print("🧠 [1/4] Running Local Vision AI...")
        sample_turbo_frames(in_path, out_path, target_frames=8)
        vision_response = local_client.chat.completions.create(
            model="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an exercise classifier. "
                            "Look at this lifting video and respond with exactly one token, no spaces: "
                            "one of: squat, bench_press, deadlift. "
                            "If unsure, pick the closest of these three."
                        ),
                    },
                    {"type": "video_url", "video_url": {
                        "url": f"file://{out_path}"}}
                ]
            }],
        )

        raw_exercise = vision_response.choices[0].message.content.strip()
        ex_key = raw_exercise if raw_exercise in MetricsAnalyzer.EXERCISE_ANGLES else "squat"
        print(f"🔍 Nemotron raw_exercise: {raw_exercise!r}")
        print(f"🏷️  Using exercise key: {ex_key}")

        # STEP 2: MEDIAPIPE MATH
        print("📐 [2/4] Running MediaPipe Engine...")
        detector = PoseDetector()
        poses, video_info = detector.process_video(in_path, sample_every_n=2)
        analyzer = MetricsAnalyzer()
        analysis_obj = analyzer.analyze(
            poses, exercise_type=ex_key, exercise_confidence=0.9)
        mp_data = dataclasses.asdict(analysis_obj)
        print("\n📹 Video info:", video_info)
        print(f"🧍 Pose samples: {len(poses)} (sample_every_n=2)")
        print(f"📊 Analysis: ex_type={analysis_obj.exercise_type}, "
              f"total_reps={analysis_obj.total_reps}, "
              f"avg_rom={analysis_obj.avg_rom:.1f}, "
              f"technique_score={analysis_obj.technique_score:.1f}, "
              f"set_rpe={analysis_obj.estimated_rpe:.1f}")
        print("📝 Notes:")
        for note in analysis_obj.notes:
            print("   -", note)

        # STEP 3: CLOUD TEXT (NVIDIA NIM)
        print("☁️ [3/4] Hitting NVIDIA NIM...")
        clean_data = mp_data.copy()
        clean_data.pop("angle_curves", None)

        # Updated prompt to force specific arrays
        sys_prompt = """
        You are an elite powerlifting coach writing FORM FEEDBACK for barbell squat, bench press, and deadlift.
        
        You will receive a JSON object with:
        - exercise_type: "squat" | "bench_press" | "deadlift"
        - total_reps, estimated_rpe, technique_score (0-100)
        - avg_rom, fatigue_index, symmetry_avg
        - reps: per-rep ROM, symmetry_score, smoothness_score, status
        - notes: model-generated technique notes (may be noisy)
        
        Your goals:
        1) Give specific, technically accurate coaching advice for THIS lift.
        2) Highlight 2–5 concrete positives ("pros") tied to the metrics.
        3) Highlight 2–5 specific corrections ("corrections") with clear cues.
        
        Rules:
        - Tailor advice to exercise_type:
          - Squat: stance, bracing, depth, knee/hip path, bar path.
          - Bench_press: setup, bar path, elbow position, leg drive, touch point.
          - Deadlift: start position, hip height, back tightness, bar path, lockout.
        - Use the metrics:
          - Low avg_rom or ROM notes → mention depth/range issues.
          - Low symmetry_avg or asymmetry notes → mention left/right control, unilateral work.
          - Low smoothness or “jerky” notes → mention tempo, control, bracing.
          - High fatigue_index or failure reps → mention load selection, set length, RPE.
        - Be concise, practical, and avoid generic phrases like "room for improvement" or "focus on form".
        - Pros should read like "Kept torso upright and stable out of the hole" not "Good form".
        - Corrections should read like clear cues, e.g. "Brace hard before each rep and keep ribs down" or "Drive knees out over midfoot on the way down".
        
        Respond STRICTLY in JSON matching this exact schema:
        {
          "advice": "string (2–4 sentences of overall feedback, referencing metrics and exercise_type)",
          "pros": ["array of 2–5 specific positives"],
          "corrections": ["array of 2–5 specific, actionable corrections/cues"]
        }
        """
        nim_response = nvidia_client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(clean_data)}
            ],
            response_format={"type": "json_object"}
        )
        ai_feedback = json.loads(nim_response.choices[0].message.content)

        # STEP 4: ANNOTATION (Save locally for debugging)
        # STEP 4: ANNOTATION (Save locally for debugging)
        if req.generateVideo:
            print(f"🎬 [4/4] Generating Annotated Video at {annotated_path}...")
            annotator = VideoAnnotator(pose_detector=detector)
            annotator.annotate_video(
                input_path=in_path,
                output_path=annotated_path,
                poses=poses,
                exercise_type=ex_key,
                reps=analysis_obj.reps,
                rpe=analysis_obj.estimated_rpe,
                technique_score=analysis_obj.technique_score
            )
        else:
            print("⏭️ [4/4] Skipping Annotated Video generation...")
        total_time = time.time() - overall_start

        # --- NEW DETAILED TERMINAL PRINTS ---
        print(
            f"\n✅ [COMPLETE] Total Time: {total_time:.2f}s. Video: {annotated_path}")
        print("\n🏋️ --- REP-BY-REP BREAKDOWN ---")
        for r in mp_data.get("reps", []):
            icon = "✅" if r["status"] == "completed" else "❌"
            print(
                f"  {icon} Rep {r['rep_number']:<2} | ROM: {r['range_of_motion']:>5.1f}° | Time: {r['duration_s']:>4.2f}s | Status: {r['status'].upper()}")
        print("----------------------------------")

        # FINAL JSON RESPONSE (Flattened for Frontend)
        final_payload = {
            "reps": mp_data.get("total_reps", 0),
            "rpe": mp_data.get("estimated_rpe", 0),
            "advice": ai_feedback.get("advice", ""),
            "score": mp_data.get("technique_score", 0),
            "pros": ai_feedback.get("pros", []),
            "corrections": ai_feedback.get("corrections", [])
        }

        print("\n📊 --- FINAL PAYLOAD SENT TO FRONTEND ---")
        print(json.dumps(final_payload, indent=2))
        print("-----------------------------------------\n")

        return final_payload

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
