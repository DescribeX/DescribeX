# DescribeX — Video Captioning Agent & Web App (Fireworks AI Edition)

This repository contains the codebase for **DescribeX**, built for the AMD Developer Hackathon (Track 2: Video Captioning Agent). It is configured to run using vision-language models on the **Fireworks AI** platform (e.g. Llama 3.2 Vision).

## Repository Structure

```
describex/
├── describex-agent/     # Hackathon Docker Image files (OpenCV Frame Extraction + Fireworks)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
├── describex-backend/   # FastAPI Backend (Render deployment)
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
└── describex-frontend/  # Vite + React Frontend (Vercel deployment)
    ├── index.html
    ├── src/
    └── package.json
```

---

## 1. Hackathon Agent (`describex-agent`)

The agent is designed to run inside a container during hackathon evaluation. It reads tasks from `/input/tasks.json`, downloads the video, extracts 10 uniformly spaced frames using OpenCV, sends them to Fireworks AI models, and outputs results to `/output/results.json`.

### How to Run Locally

1. Create an input directory and a dummy `tasks.json`:
   ```bash
   mkdir input output
   echo '[{"task_id": "v1", "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4", "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}]' > input/tasks.json
   ```

2. Build the Docker image:
   ```bash
   cd describex-agent
   docker build -t describex-agent:latest .
   ```

3. Run the container:
   ```bash
   docker run --rm \
     -v "$(pwd)/input:/input" \
     -v "$(pwd)/output:/output" \
     -e FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY" \
     describex-agent:latest
   ```

4. Check output:
   ```bash
   cat output/results.json
   ```

---

## 2. Backend Server (`describex-backend`)

The backend is built in FastAPI and uses Server-Sent Events (SSE) to bypass the Render Free Tier 30-second timeout.

### How to Run Locally

1. Install dependencies:
   ```bash
   cd describex-backend
   pip install -r requirements.txt
   ```

2. Set your API Key:
   ```bash
   # On Windows (PowerShell)
   $env:FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY"
   
   # On macOS/Linux
   export FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY"
   ```

3. Start the server:
   ```bash
   python main.py
   ```
   The backend will be running at `http://localhost:8000`.

---

## 3. Frontend Web App (`describex-frontend`)

A premium React application built with Vite and vanilla CSS.

### How to Run Locally

1. Install dependencies:
   ```bash
   cd describex-frontend
   npm install
   ```

2. Start the dev server:
   ```bash
   npm run dev
   ```
   The frontend will be running at `http://localhost:5173`.
