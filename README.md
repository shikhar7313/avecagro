# Agri Nova — Dev Setup, Services, and Safety Notes

Agri Nova bundles a marketing/landing experience, a multi-app web stack, questionnaire + weather backends, and supporting ML utilities. This guide explains what each service does, how to run it (Windows/PowerShell), and the safety/legal notes you must follow.

## Architecture at a glance
- **Landing (Vite)**: Marketing + lead capture + interactive assessment UI.
- **Questionnaire API**: Persists farmer inputs and serves recommendations.
- **Weather hook server (Python)**: Supplies local weather/soil signals into the assessment flow.
- **Node backend (rooms/tasks/etc.)**: Real-time/task features for the broader app shell.
- **React client (separate app)**: Standalone UI that pairs with the Node backend.
- **ML helpers (Python)**: Chat/data utilities and the Shikhar app for inference/demos.

## Data and compliance
- **Datasets are intentionally removed from this repo** due to licensing and regulatory constraints. You must supply your own compliant datasets and ensure you have rights to use them.
- Do not reintroduce third-party data without verifying licenses and privacy requirements. When in doubt, keep data out of the repo and load it securely at runtime.

## Legal / copyright warning
This codebase and its contents are proprietary to the contributors/organization. Unauthorized copying, distribution, or use of the code, assets, or any non-public documentation is strictly prohibited. External datasets or assets must be used only if you have explicit rights and must not be committed to the repository. Proceed only if you have permission.

## Prerequisites
- Node.js 18+ and npm
- Python 3.10+ with the repo virtualenv at `D:\Gear\agri fronti\.venv`
- Install dependencies at root (`npm install` inside `Agri_nova`) and per service where needed (client/server covered by commands below)

## Core run commands (8)
Run these in separate terminals as needed:

1. **Root web stack (Landing + Dashboard wrapper)**
   ```powershell
   Set-Location "d:\Gear\agri fronti\ansh\agri fronti\Agri_nova"; npm run start
   ```
2. **Landing (Vite) dev server**
   ```powershell
   Set-Location "D:\Gear\agri fronti\ansh\agri fronti\Agri_nova\src\Landing page\New folder"; npm run dev -- --port 6001
   ```
3. **Questionnaire API server**
   ```powershell
   Set-Location "d:\Gear\agri fronti\ansh\agri fronti\Agri_nova\src\Landing page\New folder"; npm.cmd run questionnaire-server
   ```
4. **Node backend (rooms/tasks/etc.)**
   ```powershell
   cd "d:\Gear\agri fronti\server"; npm start
   ```
5. **React client (separate app)**
   ```powershell
   cd "D:\Gear\agri fronti\client"
   npm install
   npm start
   ```
6. **Chat data helper (Python)**
   ```powershell
   & "D:/Gear/agri fronti/.venv/Scripts/python.exe" "d:/Gear/agri fronti/ansh/crop recommendation/data/chat.py"
   ```
7. **Weather hook server (Python)**
   ```powershell
   & "D:/Gear/agri fronti/.venv/Scripts/python.exe" "d:/Gear/agri fronti/ansh/agri fronti/Agri_nova/src/Landing page/New folder/src/hooks/weather_server.py"
   ```
8. **Shikhar app (Python stream/app entry)**
   ```powershell
   Set-Location "D:\Gear\agri fronti"; & "D:\Gear\agri fronti\.venv\Scripts\python.exe" "D:\Gear\agri fronti\Shikhar\app.py"
   ```

## Tips
- Keep each long-running service in its own terminal to avoid blocking.
- If ports clash, adjust `--port` flags or local env files accordingly.
- Ensure the `.venv` is active or reference its full path as shown above when running Python scripts.
- For fresh installs, run `npm install` at `Agri_nova` root and inside `src/Landing page/New folder` if dependencies change.

## Troubleshooting
- **Port in use**: Change the port flag (`--port 6002`, etc.) or stop the conflicting process.
- **Module not found**: Re-run `npm install` in the relevant folder or check that the virtualenv path is correct.
- **Python SSL/requests issues**: Update `pip`, ensure certs are available, and rerun inside the `.venv`.
