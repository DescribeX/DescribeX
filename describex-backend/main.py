"""
DescribeX Backend — FastAPI server with SSE streaming for real-time pipeline progress.
Uses the same two-stage pipeline as the agent:
  Stage 1: Dense scene description from sampled frames (Vision model)
  Stage 2: Style-specific caption generation from description (Text model)
Falls back to Mock Mode if no FIREWORKS_API_KEY is set.
"""

import os
import uuid
import json
import asyncio
import logging
import base64
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests
import cv2

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("describex-backend")

def load_env():
    # Look for .env in current, parent, or grandparent directories
    for path in [".env", "../.env", "../../.env"]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            os.environ[k.strip()] = v.strip().strip('"').strip("'")
                logger.info(f"Loaded environment variables from: {os.path.abspath(path)}")
                break
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")

load_env()

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="DescribeX Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory task store
tasks_db = {}

# ─── Fireworks Configuration ─────────────────────────────────────────────────
api_key = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

VISION_MODELS = [
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/kimi-k2p5",
]
TEXT_MODELS = [
    "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "accounts/fireworks/models/deepseek-v4-pro",
]

if api_key:
    logger.info("Fireworks API key configured. Running in REAL mode.")
else:
    logger.warning("No FIREWORKS_API_KEY found. Running in MOCK mode.")

# ─── Prompts (same as agent) ─────────────────────────────────────────────────
STAGE1_PROMPT = """You are a professional video analyst. I am showing you {n} uniformly sampled frames from a short video clip (in chronological order).

Provide a DETAILED scene description covering:
1. **Setting & Environment**: Location type (indoor/outdoor/urban/nature), weather, time of day, lighting
2. **Subjects & Objects**: What people, animals, or main objects are present; their appearance, actions, positions
3. **Movement & Action**: What is happening over time across these frames; any changes, motion, or activity
4. **Text & Signage**: Any visible text, signs, logos, or branding in the frames
5. **Mood & Atmosphere**: The overall feel — calm, energetic, professional, playful, etc.
6. **Notable Details**: Colors, textures, patterns, anything distinctive

Be thorough and factual. This description will be used to generate captions, so accuracy is critical. Write 150-250 words."""

STAGE2_PROMPT = """Based on this video description, generate captions in exactly four styles.

VIDEO DESCRIPTION:
{description}

STYLES REQUIRED:
1. **formal**: Professional, objective, factual tone. Describe what the video shows as if writing for a documentary or news broadcast. Use precise, measured language.
2. **sarcastic**: Dry, ironic, lightly mocking tone. Find something amusing or underwhelming about the scene and comment on it with subtle wit. Don't be mean-spirited.
3. **humorous_tech**: Write a funny caption that incorporates technology, programming, or software engineering references/metaphors. Connect the video content to tech concepts (debugging, APIs, git, frameworks, etc.) in a clever way.
4. **humorous_non_tech**: Write a funny, relatable, everyday humor caption. No technical jargon at all. Think observations a comedian would make about the scene.

RULES:
- Each caption should be 2-4 sentences (40-120 words)
- Captions must accurately reflect the VIDEO DESCRIPTION content
- Each style must feel distinctly different in tone
- Return ONLY a valid JSON object with exactly these four keys: formal, sarcastic, humorous_tech, humorous_non_tech"""


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"name": "DescribeX API", "version": "2.0", "mode": "REAL" if api_key else "MOCK"}

@app.get("/api/config")
def get_config():
    """Return API key for agent containers that call back to the backend."""
    return {"FIREWORKS_API_KEY": api_key}

@app.post("/api/upload")
async def upload_video(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None)
):
    task_id = str(uuid.uuid4())

    if file:
        filename = f"{task_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        try:
            with open(filepath, "wb") as buf:
                content = await file.read()
                buf.write(content)
            logger.info(f"Saved upload: {filename}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Write failed: {e}")

        tasks_db[task_id] = {
            "type": "file", "filepath": filepath,
            "filename": file.filename, "status": "pending"
        }
        return {"task_id": task_id, "filename": file.filename}

    elif video_url:
        tasks_db[task_id] = {
            "type": "url", "url": video_url, "status": "pending"
        }
        return {"task_id": task_id, "video_url": video_url}

    else:
        raise HTTPException(status_code=400, detail="No video file or URL provided.")


# ─── Frame Extraction ────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int = 8) -> List[str]:
    """Extract uniformly spaced frames, skipping intro/outro black frames."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Skip first/last 5% to avoid black/title screens
    start = int(total * 0.05)
    end = int(total * 0.95)
    usable = end - start
    if usable < num_frames:
        start, end, usable = 0, total, total

    step = usable / num_frames
    indices = [int(start + i * step) for i in range(num_frames)]

    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        max_dim = 768
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames_b64.append(base64.b64encode(buf).decode('utf-8'))

    cap.release()
    logger.info(f"Extracted {len(frames_b64)} frames from {video_path}")
    return frames_b64



def parse_captions_json(text: str) -> Optional[dict]:
    # Pre-cleanup: remove reasoning traces and markdown code block backticks if present
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try normal json.loads first
    try:
        return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"Standard JSON parse failed: {e}. Trying regex extraction...")

    # Fallback: regex extraction
    keys = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
    result = {}
    import re
    
    # Try finding each key-value pair using a regex that captures everything inside the quotes
    for k in keys:
        pattern = rf'"{k}"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*\}}|\s*$)'
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            val = match.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
            result[k] = val
        else:
            pattern_alt = rf'[\'"]?{k}[\'"]?\s*:\s*[\'"](.*?)[\'"](?=\s*,\s*[\'"]|\s*\}}|\s*$)'
            match_alt = re.search(pattern_alt, cleaned, re.DOTALL)
            if match_alt:
                val = match_alt.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
                result[k] = val
                
    if all(k in result for k in keys):
        return result
    return None

# ─── Fireworks API Helper ────────────────────────────────────────────────────

def call_fireworks(model: str, messages: list,
                   temperature: float = 0.4, max_tokens: int = 1024,
                   json_mode: bool = False, timeout: int = 60) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(FIREWORKS_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ─── SSE Generator ───────────────────────────────────────────────────────────

async def sse_generator(task_id: str):
    task = tasks_db.get(task_id)
    if not task:
        yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
        return

    def sse(step: int, msg: str, status: str, result: Optional[dict] = None):
        return f"data: {json.dumps({'step': step, 'message': msg, 'status': status, 'result': result})}\n\n"

    filepath = None

    try:
        if not api_key:
            # ── MOCK MODE ────────────────────────────────────────────────
            logger.info(f"[{task_id}] Mock Mode")
            yield sse(1, "Uploading video...", "running")
            await asyncio.sleep(1.5)
            yield sse(1, "Uploading video...", "completed")

            yield sse(2, "Extracting frames...", "running")
            await asyncio.sleep(1.5)
            yield sse(2, "Extracting frames...", "completed")

            yield sse(3, "Analyzing scene (Stage 1)...", "running")
            await asyncio.sleep(2)
            yield sse(3, "Analyzing scene (Stage 1)...", "completed")

            yield sse(4, "Generating styled captions (Stage 2)...", "running")
            await asyncio.sleep(2)
            yield sse(4, "Generating styled captions (Stage 2)...", "completed")

            yield sse(5, "Finalizing...", "running")
            await asyncio.sleep(1)
            yield sse(5, "Finalizing...", "completed")

            mock_name = task.get("filename", task.get("url", "video.mp4"))
            mock_captions = {
                "formal": (
                    f"The footage presents a medium, eye-level shot of a setting. "
                    f"The camera remains relatively stable, capturing the subject's "
                    f"environment and activity with clear, professional lighting "
                    f"under natural illumination. The composition suggests a "
                    f"deliberate and controlled production approach."
                ),
                "sarcastic": (
                    "Oh wow, another groundbreaking video file that has been "
                    "successfully uploaded to the cloud. I'm absolutely stunned "
                    "by the raw dramatic tension of these frames. Riveting stuff, "
                    "truly. Someone alert the Academy."
                ),
                "humorous_tech": (
                    "This video is basically a live-streamed CI/CD pipeline, "
                    "pushing commits of frames directly into our VLM. The "
                    "deep-learning layers are doing a git push --force on our "
                    "attention, smearing edges until the screen looks like a "
                    "CSS glitch in production. At least the aspect ratio didn't "
                    "throw a NaN."
                ),
                "humorous_non_tech": (
                    "It's like watching paint dry, except the paint is digital, "
                    "and someone is charging us money for it. At least the "
                    "colors are nice, and nothing caught on fire during the "
                    "viewing process. My cat was more entertained than I was."
                ),
            }
            yield sse(6, "completed", "completed", mock_captions)

        else:
            # ── REAL MODE — Two-Stage Pipeline ───────────────────────────
            logger.info(f"[{task_id}] Real Mode — Two-Stage Pipeline")
            loop = asyncio.get_event_loop()

            # Step 1: Acquire video file
            yield sse(1, "Uploading video...", "running")
            if task["type"] == "file":
                filepath = task["filepath"]
            else:
                temp_name = f"temp_{task_id}.mp4"
                filepath = os.path.join(UPLOAD_DIR, temp_name)
                def _download():
                    r = requests.get(task["url"], stream=True, timeout=60)
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                await loop.run_in_executor(None, _download)
                task["filepath"] = filepath
            yield sse(1, "Uploading video...", "completed")

            # Step 2: Extract frames
            yield sse(2, "Extracting frames...", "running")
            frames_b64 = await loop.run_in_executor(
                None, lambda: extract_frames(filepath, 8)
            )
            if not frames_b64:
                raise RuntimeError("Failed to extract frames from video.")
            yield sse(2, f"Extracted {len(frames_b64)} frames", "completed")

            # Step 3: Stage 1 — Scene Description (Vision Model)
            yield sse(3, "Analyzing scene (Stage 1)...", "running")

            content_parts = [
                {"type": "text", "text": STAGE1_PROMPT.format(n=len(frames_b64))}
            ]
            for frame in frames_b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                })

            description = None
            for vmodel in VISION_MODELS:
                try:
                    logger.info(f"[{task_id}] Stage 1 trying: {vmodel}")
                    result = await loop.run_in_executor(
                        None,
                        lambda m=vmodel: call_fireworks(
                            m, [{"role": "user", "content": content_parts}],
                            temperature=0.3, max_tokens=600, timeout=45
                        )
                    )
                    description = result["choices"][0]["message"]["content"].strip()
                    logger.info(f"[{task_id}] Stage 1 OK via {vmodel} ({len(description)} chars)")
                    break
                except Exception as e:
                    logger.warning(f"[{task_id}] Stage 1 {vmodel} failed: {e}")

            if not description:
                raise RuntimeError("Stage 1 (scene description) failed on all models.")
            yield sse(3, "Scene analysis complete", "completed")

            # Step 4: Stage 2 — Styled Captions (Text Model)
            yield sse(4, "Generating styled captions (Stage 2)...", "running")

            style_messages = [
                {
                    "role": "system",
                    "content": "You are a creative caption writer. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": STAGE2_PROMPT.format(description=description)
                }
            ]

            captions_data = None
            for tmodel in TEXT_MODELS:
                try:
                    logger.info(f"[{task_id}] Stage 2 trying: {tmodel}")
                    result = await loop.run_in_executor(
                        None,
                        lambda m=tmodel: call_fireworks(
                            m, style_messages,
                            temperature=0.7, max_tokens=1528,
                            json_mode=True, timeout=30
                        )
                    )
                    text = result["choices"][0]["message"]["content"].strip()
                    captions_data = parse_captions_json(text)

                    # Validate all 4 keys exist with content
                    required = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
                    missing = [k for k in required if not captions_data.get(k)]
                    if missing:
                        logger.warning(f"[{task_id}] Stage 2 {tmodel}: missing keys {missing}")
                        captions_data = None
                        continue

                    logger.info(f"[{task_id}] Stage 2 OK via {tmodel}")
                    break
                except Exception as e:
                    logger.warning(f"[{task_id}] Stage 2 {tmodel} failed: {e}")

            if not captions_data:
                raise RuntimeError("Stage 2 (caption generation) failed on all models.")
            yield sse(4, "Styled captions generated", "completed")

            # Step 5: Finalize
            yield sse(5, "Finalizing...", "running")
            await asyncio.sleep(0.3)
            yield sse(5, "Finalizing...", "completed")

            # Step 6: Deliver results
            yield sse(6, "completed", "completed", captions_data)

    except Exception as e:
        logger.error(f"[{task_id}] Pipeline error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    finally:
        fp = task.get("filepath") if task else None
        if fp and os.path.exists(fp):
            try:
                os.remove(fp)
                logger.info(f"Cleaned up: {fp}")
            except Exception:
                pass


@app.get("/api/status/{task_id}")
async def get_status_stream(task_id: str):
    return StreamingResponse(sse_generator(task_id), media_type="text/event-stream")


# ─── Entrypoint ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
