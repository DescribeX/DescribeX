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

Before submitting, test the container locally on your computer with a sample task:

#### For Windows (PowerShell):
```powershell
# 1. Create temporary folders
New-Item -ItemType Directory -Force -Path input, output

# 2. Create the test tasks.json file
Set-Content -Path input/tasks.json -Value '[{"task_id": "v1", "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4", "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}]'

# 3. Run the container mapping volumes
docker run --rm --env FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY" -v "${PWD}/input:/input" -v "${PWD}/output:/output" describex-agent:latest
```

#### For Linux / macOS (Bash):
```bash
# 1. Create folders and task file
mkdir -p input output
echo '[{"task_id": "v1", "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4", "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}]' > input/tasks.json

# 2. Run the container mapping volumes
docker run --rm --env FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY" -v "$(pwd)/input:/input" -v "$(pwd)/output:/output" describex-agent:latest
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
# DescribeX: Two-Stage Video Captioning Agent

Welcome to **DescribeX**, a professional-grade web application and command-line agent designed to generate high-quality, multi-style video captions. The system leverages a robust **Two-Stage (Describe-then-Style) Pipeline** powered by the latest **multimodal models (Kimi K2.6)** and **reasoning models (DeepSeek-V4-Pro)** via the Fireworks AI API.

---

## 🚀 Key Achievements

1. **Two-Stage Pipeline (Describe-then-Style)**: 
   - **Stage 1 (Vision)**: Extracts 8 keyframes from the middle 90% of the video and feeds them to **Kimi K2.6** to generate a dense, accurate description of the scene.
   - **Stage 2 (Text)**: Feeds the dense description to **DeepSeek-V4-Pro** to generate four distinct, tone-aligned caption styles (Formal, Sarcastic, Humorous Tech, Humorous Non-Tech).
2. **Robust JSON Parsing**: Implemented a custom `parse_captions_json` helper that uses regex fallback matching to gracefully handle and repair unescaped quotes or markdown code blocks returned by creative text models.
3. **Local Env Security**: Configured a root-level `.env` file loader globally ignored by `.gitignore` to securely manage `FIREWORKS_API_KEY` without committing secrets.
4. **Premium Frontend UI**: Built a responsive, modern React dashboard utilizing a glassmorphic dark theme, progress checklists, status bars, and interactive/editable caption cards with character counters and copy functionality.

---

## 📸 System Execution Showcase

Below is the visual confirmation of the pipeline running in **REAL mode** using a sample urban boulevard autumn video clip:

### 1. Analysis Status (Top Section)
The video was analyzed, and both stages successfully completed:
![Pipeline Success Top](C:/Users/Varsha Singh/.gemini/antigravity/brain/ab1632aa-7279-4920-9379-22f4007dfefc/pipeline_success_top_1783695789328.png)

### 2. Generated Caption Cards (Bottom Section)
Four style-aligned captions with individual copy actions and editable text fields:
![Pipeline Success Bottom](C:/Users/Varsha Singh/.gemini/antigravity/brain/ab1632aa-7279-4920-9379-22f4007dfefc/pipeline_success_bottom_1783695795984.png)

---

## 📝 Generated Captions Example

| Style | Caption Content |
| :--- | :--- |
| **Formal** | *This time-lapse sequence captures the rhythmic flow of traffic along a sunlit urban boulevard during autumn's golden hour. Vehicles blur into continuous streaks of light, conveying the ceaseless movement of a busy city. Lining the avenue, trees display vivid yellow foliage, while distant mountains and high-rise apartments frame the scene. A building bearing the sign 'Korea Military Engineering' anchors the composition, adding a distinct sense of place.* |
| **Sarcastic** | *Behold the cinematic masterpiece: cars driving on a road. The time-lapse effect elevates this mundane commute into something almost profound, if you squint. Autumn leaves gamely try to distract from the endless procession of buses and sedans, but the real star is that 'Korea Military Engineering' building, presumably engineering the very concept of sitting in traffic.* |
| **Humorous Tech** | *Someone set the city's traffic simulation to 10x speed and forgot to disable motion blur. Each vehicle is a data packet racing along the arterial highway, with the 'Korea Military Engineering' office likely debugging the routing algorithm that caused this golden-hour gridlock. The traffic lights are green, but the latency is still terrible—typical legacy infrastructure.* |
| **Humorous Non-Tech** | *This time-lapse proves that no matter how golden the hour or how pretty the autumn leaves, traffic is still just a parade of other people ruining your day. I love that a building labeled 'Korea Military Engineering' is right there, as if they're in charge of deploying the world's most aggressive rush hour. Those blurred cars aren't moving fast—they're just vibrating with impatience.* |

---

## 🛠️ Project Structure

```
describex/
├── .env                  <-- Git-ignored API key storage
├── .gitignore            <-- Ignores local secrets
├── describex-backend/    <-- FastAPI server running on port 8000
│   ├── main.py           <-- SSE streaming pipeline and API routes
│   └── requirements.txt
├── describex-frontend/   <-- React/Vite app running on port 5173
│   ├── src/
│   │   ├── App.jsx       <-- Glassmorphic user interface
│   │   └── App.css       <-- CSS Design tokens and layout rules
│   └── package.json
└── describex-agent/      <-- Offline execution CLI agent (Dockerized)
    ├── main.py           <-- Batch worker script
    └── Dockerfile
```

---

## ⚙️ Running Locally

### 1. Set the API Key
Create a `.env` file at the root of the workspace:
```env
FIREWORKS_API_KEY="fw_YOUR_KEY_HERE"
```

### 2. Start the Backend
```bash
cd describex-backend
pip install -r requirements.txt
python main.py
```

### 3. Start the Frontend
```bash
cd describex-frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🐙 1. Push Code to GitHub

Follow these steps to initialize your git repository and push all source files to GitHub:

```bash
# 1. Initialize git (if not already done)
git init

# 2. Stage all modifications (the root-level .env is automatically ignored)
git add -A

# 3. Commit the changes
git commit -m "feat: complete two-stage video captioning system with Kimi and DeepSeek"

# 4. Create a main branch and link to your GitHub repository
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

# 5. Push the code
git push -u origin main
```

---

## 🐳 2. Build & Push Docker Agent (Track 2 Submission)

Your agent container must execute autonomously, read tasks from `/input/tasks.json`, and output results to `/output/results.json` within 10 minutes.

### Step A: Build & Tag the Container
Navigate to the agent folder and build the image locally:
```bash
cd describex-agent
docker build -t describex-agent:latest .
```

Tag it using the GitHub Container Registry (GHCR) format (replace `YOUR_GITHUB_USERNAME` with your actual GitHub handle):
```bash
docker tag describex-agent:latest ghcr.io/YOUR_GITHUB_USERNAME/describex-agent:latest
```

### Step B: Run & Test Locally
Before submitting, test the container locally with a sample task:

#### For Windows (PowerShell):
```powershell
# 1. Create temporary folders
New-Item -ItemType Directory -Force -Path input, output

# 2. Create the test tasks.json file
Set-Content -Path input/tasks.json -Value '[{"task_id": "v1", "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4", "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}]'

# 3. Run the docker container mapping volumes
docker run --rm -v "${PWD}/input:/input" -v "${PWD}/output:/output" describex-agent:latest
```

#### For Linux / macOS (Bash):
```bash
# 1. Create folders and task file
mkdir -p input output
echo '[{"task_id": "v1", "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4", "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}]' > input/tasks.json

# 2. Run the container mapping volumes
docker run --rm -v "$(pwd)/input:/input" -v "$(pwd)/output:/output" describex-agent:latest
```
Check `output/results.json` to verify the generated captions.

### Step C: Log In & Push to GHCR
Log in to GitHub Container Registry using a GitHub Personal Access Token (PAT) with `write:packages` scope:
```bash
echo "YOUR_GITHUB_PAT" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

Push your image to the public registry:
```bash
docker push ghcr.io/YOUR_GITHUB_USERNAME/describex-agent:latest
```

> [!IMPORTANT]
> **Visibility Setting**: Go to your GitHub profile → **Packages** → **describex-agent** → **Package Settings** and change visibility to **Public** so that the judging system can pull the image.

**Your Final Submission URI**: `ghcr.io/YOUR_GITHUB_USERNAME/describex-agent:latest`

---

## ☁️ 3. Deploy Backend to Render

1. Sign in to your [Render Dashboard](https://dashboard.render.com).
2. Click **New** -> **Web Service**.
3. Connect your GitHub repository.
4. Set the following configuration:
   - **Name**: `describex-backend`
   - **Root Directory**: `describex-backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
5. Click **Advanced** and add the following Environment Variables:
   - `FIREWORKS_API_KEY`: `fw_YOUR_KEY_HERE` (your actual Fireworks key)
   - `PORT`: `8000`
6. Click **Deploy Web Service**.
7. Once deployed, copy your backend service URL (e.g., `https://describex-backend.onrender.com`).

---

## ⚡ 4. Deploy Frontend to Vercel

1. Sign in to your [Vercel Dashboard](https://vercel.com).
2. Click **Add New** -> **Project** and import your GitHub repository.
3. Configure the following project settings:
   - **Root Directory**: `describex-frontend`
   - **Framework Preset**: `Vite`
4. Expand **Environment Variables** and add:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://describex-backend.onrender.com` *(Replace this with your actual Render backend URL)*
5. Click **Deploy**.

