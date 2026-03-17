# Kinara — AI Powerlifting Form Analyser

Kinara is a full-stack web application that uses computer vision and AI to analyse powerlifting technique in real time. Athletes upload a video of their lift, receive an instant form score, and get detailed coaching feedback highlighting strengths and corrections.

Built for the UCL AI Hackathon 2026. Worked as a part of a team to create this.

---

## Features

- **AI Form Analysis** — Upload any powerlifting video and receive a score (0–100), RPE estimate, rep count, pros, and corrections
- **Injury Prevention** — Identifies dangerous form errors before they become chronic injuries
- **Session History** — Every analysed lift is saved to a personal history with timestamps and scores
- **Trend Insights** — Track form score progression over time across sessions
- **Authentication** — Secure sign-up and login via Clerk
- **Dark / Light Mode** — Full theme support

---

## Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| [Next.js 16](https://nextjs.org) | React framework with App Router and API routes |
| [React 19](https://react.dev) | UI component library |
| [TypeScript](https://www.typescriptlang.org) | Static typing throughout |
| [Tailwind CSS v4](https://tailwindcss.com) | Utility-first styling |
| [Framer Motion](https://www.framer.com/motion) | Animations and transitions |
| [Lucide React](https://lucide.dev) | Icon library |

### Backend & Infrastructure
| Technology | Purpose |
|---|---|
| [Next.js API Routes](https://nextjs.org/docs/app/building-your-application/routing/route-handlers) | Serverless proxy to the ML backend |
| [Clerk](https://clerk.com) | Authentication and user management |
| [MySQL2](https://github.com/sidorares/node-mysql2) | Database client for session storage |
| [Cloudinary](https://cloudinary.com) | Video upload and storage |
| [Brev.dev (NVIDIA GPU)](https://brev.dev) | Hosts the ML inference backend |

### ML Backend (separate service)
- Computer vision model served at `POST /full-analysis`
- Accepts: `{ videoUrl: string }`
- Returns: `{ reps, rpe, score, advice, pros, corrections, liftType }`

---

## Project Structure

```
app/
  api/
    analyze-lift/     # Proxy route to ML backend
    sessions/         # CRUD for lift session history
  dashboard/          # User dashboard with stats and history
  lifts/              # General lift analysis page
  (auth)/             # Login and sign-up pages
components/
  LiftPage.tsx        # Core upload + analysis UI
  FeedbackCard.tsx    # Displays AI feedback results
  Navbar.tsx          # Navigation bar
  TrendInsight.tsx    # Score trend visualisation
lib/
  db.ts               # MySQL connection pool
  types.ts            # Shared TypeScript types
supabase/
  schema.sql          # Database schema
```

---

## Getting Started

### Prerequisites
- Node.js >= 20.9.0
- MySQL database
- Clerk account
- Cloudinary account

### Environment Variables

Create a `.env.local` file in the root:

```env
# Clerk
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=

# Cloudinary
NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME=
NEXT_PUBLIC_CLOUDINARY_UPLOAD_PRESET=

# Database
DB_HOST=
DB_USER=
DB_PASSWORD=
DB_NAME=
```

### Run Locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Production Build

```bash
npm run build
npm start
```

---

## Database Schema

See [`supabase/schema.sql`](supabase/schema.sql) for the full schema. The core table is `lift_sessions`:

```sql
CREATE TABLE lift_sessions (
  id           CHAR(36)      PRIMARY KEY DEFAULT (UUID()),
  user_id      VARCHAR(255)  NOT NULL,
  lift_type    VARCHAR(50)   NOT NULL,
  video_url    TEXT          NOT NULL,
  score        TINYINT       NOT NULL,
  pros         JSON          NOT NULL,
  corrections  JSON          NOT NULL,
  created_at   TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);
```

---

## Licence

MIT
