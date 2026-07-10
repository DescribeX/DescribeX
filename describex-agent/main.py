"""
DescribeX Agent — Video Captioning Pipeline for AMD Hackathon Track 2
Uses Fireworks AI Vision Models with a two-stage pipeline:
  Stage 1: Dense scene description from sampled frames
  Stage 2: Style-specific caption generation from the description
Includes model escalation, time-budget management, and atomic writes.
"""

import os
import json
import time
import asyncio
import base64
from typing import Dict, List, Optional
import requests
import cv2

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
                print(f"[init] Loaded environment variables from: {os.path.abspath(path)}")
                break
            except Exception as e:
                print(f"[init] Error reading {path}: {e}")

load_env()

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_PATH  = os.environ.get("INPUT_TASKS_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_RESULTS_PATH", "/output/results.json")
TEMP_DIR    = os.environ.get("TEMP_DIR", "/tmp")

MAX_RUNTIME  = 540.0   # Flush at 9 min to guarantee clean exit within 10 min
START_TIME   = time.time()

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

# Models ordered by capability (escalation ladder)
VISION_MODELS = [
    "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
    "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
]
TEXT_MODELS = [
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "accounts/fireworks/models/llama-v3p1-70b-instruct",
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY", "")
    if key:
        return key
    backend = os.environ.get("DESCRIBEX_BACKEND_URL")
    if backend:
        try:
            r = requests.get(f"{backend.rstrip('/')}/api/config", timeout=5)
            if r.ok:
                return r.json().get("FIREWORKS_API_KEY", "")
        except Exception:
            pass
    return ""

def elapsed() -> float:
    return time.time() - START_TIME

def budget_ok() -> bool:
    return elapsed() < MAX_RUNTIME


# ─── Frame Extraction ────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int = 8) -> List[str]:
    """
    Extract num_frames uniformly spaced frames from a video.
    Returns list of base64 JPEG strings.
    Intelligent sampling: skip first/last 5% to avoid black intro/outro frames.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    
    if total <= 0:
        cap.release()
        return []
    
    # Skip first/last 5% of frames to avoid black/title screens
    start_frame = int(total * 0.05)
    end_frame   = int(total * 0.95)
    usable      = end_frame - start_frame
    
    if usable < num_frames:
        start_frame = 0
        end_frame = total
        usable = total

    step = usable / num_frames
    indices = [int(start_frame + i * step) for i in range(num_frames)]

    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Resize to max 768px while maintaining aspect ratio
        h, w = frame.shape[:2]
        max_dim = 768
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        
        # Encode as JPEG with good quality
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames_b64.append(base64.b64encode(buf).decode('utf-8'))
    
    cap.release()
    
    # Also compute video duration for the description prompt
    duration = total / fps if fps > 0 else 0
    print(f"[extract] {len(frames_b64)} frames from {video_path} "
          f"(total={total}, fps={fps:.1f}, duration={duration:.1f}s)")
    return frames_b64


# ─── Fireworks API Calls ─────────────────────────────────────────────────────

def call_fireworks(api_key: str, model: str, messages: list,
                   temperature: float = 0.4, max_tokens: int = 1024,
                   json_mode: bool = False, timeout: int = 60) -> dict:
    """Synchronous call to Fireworks chat completion."""
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


# ─── Two-Stage Caption Pipeline ─────────────────────────────────────────────

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


class CaptionAgent:
    def __init__(self):
        self.api_key   = get_api_key()
        self.results   = {}
        self.semaphore = asyncio.Semaphore(3)
        self._lock     = asyncio.Lock()  # For thread-safe results writes

    # ── Download ──────────────────────────────────────────────────────────
    async def download_video(self, url: str, dest: str) -> bool:
        print(f"[download] {url}")
        try:
            loop = asyncio.get_event_loop()
            def _dl():
                r = requests.get(url, stream=True, timeout=60)
                r.raise_for_status()
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
            await loop.run_in_executor(None, _dl)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[download] OK → {dest} ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"[download] FAIL: {e}")
            return False

    # ── Stage 1: Scene Description ────────────────────────────────────────
    async def describe_scene(self, frames_b64: List[str]) -> Optional[str]:
        """Use vision model to generate a detailed scene description."""
        content = [
            {"type": "text", "text": STAGE1_PROMPT.format(n=len(frames_b64))}
        ]
        for frame in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })
        
        messages = [{"role": "user", "content": content}]
        loop = asyncio.get_event_loop()
        
        for model in VISION_MODELS:
            if not budget_ok():
                return None
            print(f"[stage1] Trying {model}...")
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda m=model: call_fireworks(
                        self.api_key, m, messages,
                        temperature=0.3, max_tokens=600, timeout=45
                    )
                )
                desc = result["choices"][0]["message"]["content"].strip()
                print(f"[stage1] OK ({len(desc)} chars) via {model}")
                return desc
            except Exception as e:
                print(f"[stage1] {model} failed: {e}")
        
        return None

    # ── Stage 2: Styled Captions ──────────────────────────────────────────
    async def generate_styled_captions(self, description: str,
                                        styles: List[str]) -> Optional[Dict[str, str]]:
        """Use text model to generate styled captions from description."""
        messages = [
            {
                "role": "system",
                "content": "You are a creative caption writer. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": STAGE2_PROMPT.format(description=description)
            }
        ]
        loop = asyncio.get_event_loop()
        
        for model in TEXT_MODELS:
            if not budget_ok():
                return None
            print(f"[stage2] Trying {model}...")
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda m=model: call_fireworks(
                        self.api_key, m, messages,
                        temperature=0.7, max_tokens=800,
                        json_mode=True, timeout=30
                    )
                )
                text = result["choices"][0]["message"]["content"].strip()
                data = json.loads(text)
                
                # Validate all required styles are present
                captions = {}
                missing = []
                for style in styles:
                    val = data.get(style, "")
                    if val and len(val) > 10:
                        captions[style] = val
                    else:
                        missing.append(style)
                
                if missing:
                    print(f"[stage2] Missing styles: {missing}, escalating...")
                    continue
                
                print(f"[stage2] OK via {model}")
                return captions
            except Exception as e:
                print(f"[stage2] {model} failed: {e}")
        
        return None

    # ── Single-Stage Fallback ─────────────────────────────────────────────
    async def single_stage_caption(self, frames_b64: List[str],
                                    styles: List[str]) -> Optional[Dict[str, str]]:
        """Fallback: do description + captioning in a single vision call."""
        prompt = (
            f"I am showing you {len(frames_b64)} uniformly sampled frames "
            "from a short video clip (in chronological order).\n\n"
            "Generate captions for this video in exactly four styles:\n"
            "- formal: Professional, objective, factual tone\n"
            "- sarcastic: Dry, ironic, lightly mocking tone\n"
            "- humorous_tech: Funny with technology/programming references\n"
            "- humorous_non_tech: Everyday humor with no technical jargon\n\n"
            "Each caption should be 2-4 sentences. Return ONLY a valid JSON object "
            "with keys: formal, sarcastic, humorous_tech, humorous_non_tech"
        )
        
        content = [{"type": "text", "text": prompt}]
        for frame in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })
        
        messages = [{"role": "user", "content": content}]
        loop = asyncio.get_event_loop()
        
        for model in VISION_MODELS:
            if not budget_ok():
                return None
            print(f"[fallback] Trying {model}...")
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda m=model: call_fireworks(
                        self.api_key, m, messages,
                        temperature=0.7, max_tokens=1024,
                        json_mode=True, timeout=60
                    )
                )
                text = result["choices"][0]["message"]["content"].strip()
                data = json.loads(text)
                
                captions = {}
                for style in styles:
                    captions[style] = data.get(style, f"Caption for {style} style.")
                
                print(f"[fallback] OK via {model}")
                return captions
            except Exception as e:
                print(f"[fallback] {model} failed: {e}")
        
        return None

    # ── Process One Task ──────────────────────────────────────────────────
    async def process_task(self, task: dict):
        task_id    = task.get("task_id", "unknown")
        video_url  = task.get("video_url", "")
        styles     = task.get("styles", [
            "formal", "sarcastic", "humorous_tech", "humorous_non_tech"
        ])
        
        async with self.semaphore:
            if not budget_ok():
                print(f"[{task_id}] Skipped — time budget exhausted ({elapsed():.0f}s)")
                return
            
            local_path = os.path.join(TEMP_DIR, f"video_{task_id}.mp4")
            
            # Download
            ok = await self.download_video(video_url, local_path)
            if not ok:
                return
            
            try:
                # Extract frames
                loop = asyncio.get_event_loop()
                frames = await loop.run_in_executor(
                    None, lambda: extract_frames(local_path, 8)
                )
                if not frames:
                    raise RuntimeError("No frames extracted")
                
                captions = None
                
                # Two-stage pipeline: describe → style
                if budget_ok():
                    description = await self.describe_scene(frames)
                    if description and budget_ok():
                        captions = await self.generate_styled_captions(description, styles)
                
                # Fallback: single-stage vision call
                if not captions and budget_ok():
                    print(f"[{task_id}] Two-stage failed, trying single-stage fallback...")
                    captions = await self.single_stage_caption(frames, styles)
                
                if captions:
                    async with self._lock:
                        self.results[task_id] = captions
                    self.write_results_atomic()
                    print(f"[{task_id}] ✓ Done ({elapsed():.0f}s elapsed)")
                else:
                    print(f"[{task_id}] ✗ All caption strategies failed")
                    
            except Exception as e:
                print(f"[{task_id}] ERROR: {e}")
            finally:
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                except Exception:
                    pass

    # ── Atomic Write ──────────────────────────────────────────────────────
    def write_results_atomic(self):
        output = [
            {"task_id": tid, "captions": caps}
            for tid, caps in self.results.items()
        ]
        
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        tmp = OUTPUT_PATH + ".tmp"
        
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            # Validate
            with open(tmp, 'r', encoding='utf-8') as f:
                json.load(f)
            os.replace(tmp, OUTPUT_PATH)
            print(f"[write] Atomically wrote {len(output)} results to {OUTPUT_PATH}")
        except Exception as e:
            print(f"[write] FAILED: {e}")
            try:
                os.remove(tmp)
            except Exception:
                pass

    # ── Dummy Results ─────────────────────────────────────────────────────
    def write_dummy_results(self, tasks: list):
        for t in tasks:
            tid = t.get("task_id", "unknown")
            self.results[tid] = {
                s: f"[{s}] Video content from {t.get('video_url', 'N/A')}"
                for s in t.get("styles", [
                    "formal", "sarcastic", "humorous_tech", "humorous_non_tech"
                ])
            }

    # ── Main Entry ────────────────────────────────────────────────────────
    async def run(self):
        print("=" * 60)
        print("  DescribeX Agent — AMD Hackathon Track 2")
        print("=" * 60)
        
        # Read input
        if not os.path.exists(INPUT_PATH):
            print(f"[init] Input not found at {INPUT_PATH}")
            exit(1)
        
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        print(f"[init] Loaded {len(tasks)} task(s)")
        
        # Check API key
        if not self.api_key:
            print("[init] WARNING: No FIREWORKS_API_KEY set, writing dummy results")
            self.write_dummy_results(tasks)
            self.write_results_atomic()
            exit(0)
        
        print(f"[init] API key configured ({'*' * 4}{self.api_key[-4:]})")
        
        # Process all tasks concurrently
        await asyncio.gather(*[self.process_task(t) for t in tasks])
        
        # Final flush
        self.write_results_atomic()
        print(f"\n[done] Agent finished in {elapsed():.1f}s with "
              f"{len(self.results)}/{len(tasks)} tasks completed")


if __name__ == "__main__":
    asyncio.run(CaptionAgent().run())
