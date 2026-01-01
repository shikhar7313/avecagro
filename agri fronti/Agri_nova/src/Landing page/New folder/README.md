# AVEC AGRO – Autonomous Farm Intelligence Landing Page

A production-style marketing site for Avec Agro, built with **React 19**, **Vite**, and **Tailwind CSS**. The layout mirrors a modern agritech SaaS experience with hero storytelling, eight-engine intelligence grid, workflow, hardware overview, validation statistics, testimonials, and bilingual footer actions.

## Tech Stack
- React 19 + TypeScript via Vite
- Tailwind CSS 3 with a custom palette (leaf, moss, brand, fern, ink) and Space Grotesk + Inter typography
- Section content is driven by data arrays in `src/App.tsx` so copy changes stay centralized
- React Router DOM powers the choose-plan → confirmation → secure payment journey

## Getting Started
1. **Install dependencies**
   ```bash
   npm install
   ```
2. **Start the user-save helper (Node)**
   ```bash
   npm run server
   ```
   The Express helper listens on `http://localhost:6002` and writes incoming payloads straight to `Assets/login.json`. Keep it running while you work on auth flows.
3. **Start the dev server**
   ```bash
   npm run dev
   ```
   The command launches Vite at `http://localhost:5173/` (add `-- --port 6001` if you want to match the dedicated preview port). If you prefer VS Code Tasks, run **Terminal → Run Task → `npm: dev`**—the task prepends the Node installation path automatically.
4. **Type-check & build**
   ```bash
   npm run build
   ```
5. **Preview the production bundle**
   ```bash
   npm run preview
   ```

## Tailwind Notes
- Base styles + reusable component classes (`.button`, `.card`, `.section`, `.hero-node`) live in `src/index.css` using `@layer components`.
- Theme tokens reside in `tailwind.config.js`; update the palette there if you rebrand the landing page.

## Key Files
- `src/App.tsx` – Hosts the multi-route experience: landing sections plus the `/confirm` and `/payment` screens that the Choose Plan flow uses.
- `src/index.css` – Tailwind imports and custom class patterns for consistent visuals.
- `.vscode/tasks.json` – Provides an `npm: dev` task that invokes `node.exe` directly to avoid Windows PATH issues.
- `Assets/Logo.png` – Brand mark rendered beside the AVEC AGRO logotype in the navbar.
- `src/data/users.json` – Seed data for the auth flow; new registrations append to in-memory state and trigger a JSON download snapshot.

## Plan Checkout Flow
- Hit the nav **Log in** button to open `/auth`, authenticate (email/password or Google), and ensure activations are tied to your profile name.
1. Selecting **Choose Plan** on any pricing card routes to `/confirm` with the plan metadata carried in router state.
2. The confirmation view surfaces inclusions, onboarding expectations, and lets you share deployment notes.
3. **Proceed to Payment** advances to `/payment`, a secure-form mock that captures payer details, surfaces plan summary, and auto-redirects to the landing page a moment after success.
4. Once a payment succeeds, that plan ID is persisted for the logged-in profile—its card shows **Activated** instead of **Choose Plan**, while other tiers remain selectable.

## Authentication Page

## Save-Users Helper API
The helper lives in `server/index.mjs` (Express + CORS). Run it with `npm run server`; it defaults to `http://localhost:6002` and persists every payload to `Assets/login.json` (creating the file if missing).
The React app POSTs to `VITE_USERS_API` (falls back to `http://localhost:6002/api/users`). Override this via `.env` if you deploy the helper elsewhere.
Endpoints:
   - `GET /health` → reports the resolved file path.
   - `POST /api/users` → expects an array of `StoredUser` entries and writes the prettified JSON to disk.

## Free Assessment Dialog
- Step 1 collects acreage, soil type, irrigation style, budget appetite, and seasonal goals.
- Submitting generates tailored crop rotation ideas in a follow-up dialog, with quick actions to tweak inputs again or close the overlay.

## Watch Demo Overlay
- Hitting **Watch Demo** in the nav or hero opens a video modal showcasing the autonomous farm command center without leaving the page.

## Deployment
Running `npm run build` outputs static assets to `dist/`. Deploy that folder to any static host (Azure Static Web Apps, Netlify, Vercel, etc.).

## Next Steps
- Replace placeholder hero visual cards with actual drone/rover renders.
- Connect CTAs to live flows (assessment form, sales scheduling).
- Localize copy blocks if you need a Hindi-first experience.
