#!/usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()   # MUST run before os.getenv()

import os
import re
import json
import time
import tempfile
import requests
from typing import Optional, Any
import subprocess
from urllib.parse import urlparse, urljoin, parse_qs
def extract_secret_from_url(url: str) -> Optional[str]:
    """
    If quiz URL contains ?secret=..., return that value.
    Otherwise return None.
    """
    try:
        q = parse_qs(urlparse(url).query)
        val = q.get("secret")
        if val and len(val) > 0:
            return val[0]
    except Exception:
        pass
    return None


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
print("Loaded STUDENT_SECRET =", os.getenv("STUDENT_SECRET"), flush=True)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
STUDENT_SECRET = os.getenv("STUDENT_SECRET")
if not STUDENT_SECRET:
    raise RuntimeError("STUDENT_SECRET env variable missing")

TOTAL_TIME_BUDGET = 170
MAX_STEPS = 20
CSV_READ_TIMEOUT = 15
official_email = "namrata@example.com"
# ---------------------------------------------------------
# API
# ---------------------------------------------------------
app = FastAPI()

class TaskRequest(BaseModel):
    email: str
    secret: str
    url: str

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def log(*a):
    print(*a, flush=True)

def ensure_abs(base_url: str, link: Optional[str]) -> Optional[str]:
    if not link:
        return None
    link = link.strip()
    if link.startswith("http://") or link.startswith("https://"):
        return link
    try:
        return urljoin(base_url, link)
    except:
        p = urlparse(base_url)
        return f"{p.scheme}://{p.netloc}{link}"

def extract_submit(html: str, page_url: str) -> Optional[str]:
    m = re.search(r"https?://[^\s\"'<>]+/submit[^\s\"'<>]*", html)
    if m:
        return m.group(0)
    m2 = re.search(
        r'["\']?\s*<span[^>]*class=["\']origin["\'][^>]*>.*?</span>\s*/submit',
        html, flags=re.IGNORECASE | re.DOTALL
    )
    if m2:
        p = urlparse(page_url)
        return f"{p.scheme}://{p.netloc}/submit"
    m3 = re.search(r'["\'](/submit[^"\'<>]*)["\']', html)
    if m3:
        return ensure_abs(page_url, m3.group(1))
    p = urlparse(page_url)
    return f"{p.scheme}://{p.netloc}/submit"

def find_csv_link(html: str, base_url: str) -> Optional[str]:
    m = re.search(r'<a[^>]+href=["\']([^"\']*\.csv)["\']', html, re.IGNORECASE)
    if m:
        return ensure_abs(base_url, m.group(1))
    m2 = re.search(r'(https?://[^\s"\']+\.csv)', html, re.IGNORECASE)
    if m2:
        return m2.group(1)
    return None

def download_to_file(url: str, timeout: int = CSV_READ_TIMEOUT) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            log("[download] HTTP", r.status_code, "for", url)
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(r.content)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception as e:
        log("[download] Exception:", e)
        return None

def read_first_column_sum(csv_path: str, cutoff: float) -> Optional[Any]:
    """
    Read only the first column of csv_path, coerce numbers (allow commas),
    filter values >= cutoff, and return sum as int when integer-valued.
    """
    try:
        # Read only first column to save memory and avoid header issues
        df = pd.read_csv(csv_path, header=0, usecols=[0], engine="python", on_bad_lines='warn')
        if df.shape[1] == 0:
            return None

        col0 = df.iloc[:, 0].astype(str).str.strip()

        # Remove common thousands separators and non-breaking spaces
        col0 = col0.str.replace(r"[,\u00A0]", "", regex=True)

        # Coerce to numeric; errors -> NaN; drop NaN
        nums = pd.to_numeric(col0, errors="coerce").dropna()

        # Filter with cutoff if provided
        if cutoff is not None:
            nums = nums[nums >= cutoff]

        if nums.empty:
            return None

        total = nums.sum()

        # If total is integer-valued, return int
        if float(total).is_integer():
            return int(total)
        return float(total)

    except Exception as e:
        log("[read_first_column_sum] Exception:", e)
        return None


def compute_cutoff_from_email(email: str) -> int:
    if not email:
        return 0
    local = email.split("@")[0]
    digits_all = re.findall(r"\d+", local)
    if digits_all:
        j = "".join(digits_all)
        try:
            if len(j) > 8:
                j = j[-6:]
            v = int(j)
            if v > 10_000_000:
                v = sum(int(x) for x in j)
            return v
        except:
            pass
    digits = re.findall(r"\d", local)
    if digits:
        s = sum(int(d) for d in digits)
        if s > 0: return s
    m = re.search(r"\d{1,2}", local)
    if m:
        try: return int(m.group(0))
        except: pass
    return 0

# ---------------------------------------------------------
# Demo solvers
# ---------------------------------------------------------
def solve_demo_page(text: str, html: str) -> str:
    return "anything you want"

def solve_demo_scrape(text: str, html: str) -> Optional[Any]:
    # Try extract numbers from visible text first, then fallback to raw html
    source = text if text and len(text) > 0 else html

    # Regex to capture numbers with optional thousands separators and decimals, e.g. -1,234.56
    nums = re.findall(r"-?\d{1,3}(?:[,\u00A0]\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", source)

    if not nums:
        return None

    # Normalize by removing commas/non-breaking spaces, then cast to float
    vals = []
    for n in nums:
        n_norm = re.sub(r"[,\u00A0]", "", n)
        try:
            if "." in n_norm:
                vals.append(float(n_norm))
            else:
                vals.append(int(n_norm))
        except:
            try:
                vals.append(float(n_norm))
            except:
                continue

    if not vals:
        return None

    total = sum(vals)

    # if all extracted tokens were integers, return int total
    if all(isinstance(v, int) for v in vals):
        return int(total)
    return float(total)


def solve_demo_audio(text: str, html: str, base_url: str, email_for_cutoff: str) -> Optional[Any]:
    # Import helpers from universal_solver
    from universal_solver import find_audio_link,dbg , tool_audio_transcribe, call_llm_audio

    audio_url = find_audio_link(html, base_url)
    if not audio_url:
        dbg("[AUDIO] No audio link found")
        return None

    # 2. Download audio as base64
    audio_blob = tool_audio_transcribe(audio_url)
    if "error" in audio_blob:
        dbg("[AUDIO] Download error:", audio_blob["error"])
        return None

    # 3. Build prompt for Groq
    prompt = (
        "Transcribe the audio EXACTLY.\n"
        "Extract ALL numbers spoken.\n"
        "Convert every spoken number to an integer.\n"
        "Return ONLY JSON in this format:\n"
        '{"numbers": [ ... ]}'
    )

    # 4. Ask Groq LLM to extract numbers
    llm_result = call_llm_audio(prompt, audio_blob["base64"])

    if not llm_result or "numbers" not in llm_result:
        dbg("[AUDIO] Failed to extract numbers:", llm_result)
        return None

    nums = llm_result["numbers"]

    # 5. Sum of spoken numbers
    total = sum(nums)
    dbg("[AUDIO] SUM OF SPOKEN NUMBERS:", total)

    return total




    log("[AUDIO] Computed answer:", answer, "cutoff:", cutoff_val)
    return answer


# ---------------------------------------------------------
# Lightweight page fetcher (replaces Playwright)
# ---------------------------------------------------------
def fetch_page(url: str) -> (bool, Optional[str], Optional[str]):
    """
    Returns tuple: (ok, html, text)
    Handles:
      - data:text/html,<html>... (URL-encoded or raw)
      - fake:// URLs with ?html=<encoded HTML>
      - normal http(s) GET
    """
    try:
        # data:text/html,<payload>
        if url.startswith("data:text/html"):
            try:
                payload = url.split(",", 1)[1]
                # payload may be URL-encoded; unquote it
                decoded = requests.utils.unquote(payload)
                html = decoded
                text = re.sub(r"<[^>]+>", "", html)
                return True, html, text
            except Exception as e:
                log("[fetch_page] data: parse error:", e)
                return False, None, None

        # fake:// scheme for tests
        if url.startswith("fake://"):
            try:
                parsed = urlparse(url)
                q = parse_qs(parsed.query)
                html_enc = q.get("html", [""])[0]
                html = requests.utils.unquote(html_enc)
                text = re.sub(r"<[^>]+>", "", html)
                return True, html, text
            except Exception as e:
                log("[fetch_page] fake:// parse error:", e)
                return False, None, None

        # Normal HTTP(s)
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                html = r.text
                text = re.sub(r"<[^>]+>", "", html)[:10000]
                return True, html, text
            else:
                log("[fetch_page] HTTP status", r.status_code, "for", url)
                return False, None, None
        except Exception as e:
            log("[fetch_page] requests.get error:", e)
            return False, None, None

    except Exception as e:
        log("[fetch_page] Unexpected error:", e)
        return False, None, None

# ---------------------------------------------------------
# STEP PROCESSOR
# ---------------------------------------------------------
def process_step(quiz_url: str, email: str, secret: str) -> Optional[str]:
    log("[STEP] Visiting", quiz_url)

    # Debug: show any secret param included in the quiz URL
    parsed_q = parse_qs(urlparse(quiz_url).query)
    log("[DEBUG] URL secret param:", parsed_q.get("secret"))
    log("[DEBUG] Using STUDENT_SECRET:", STUDENT_SECRET)

    # Use secret embedded in the URL if present (the quiz host can provide it per-step)
    secret_from_url = extract_secret_from_url(quiz_url)
    if secret_from_url:
        real_secret = secret_from_url
    else:
        # fallback to canonical env secret
        real_secret = STUDENT_SECRET


    html = ""
    text = ""

    # Use lightweight fetcher (handles data: and fake: and normal http)
    ok, html, text = fetch_page(quiz_url)
    if not ok:
        log("[FETCH] Could not fetch page", quiz_url)
        return None

    # include form actions if present (best-effort)
    try:
        # find simple form action attributes without executing JS
        for m in re.finditer(r'<form[^>]+action=["\']([^"\']+)["\']', html, re.IGNORECASE):
            a = m.group(1)
            if a:
                html += f"\n<!-- form action: {a} -->"
    except Exception:
        pass

    submit_candidate = extract_submit(html, quiz_url)
    log("[STEP] Candidate submit url:", submit_candidate)

    path = urlparse(quiz_url).path

    answer = None

    # --- FIXED DEMO HANDLING (PREVENTS INTERFERENCE WITH REAL QUIZZES) ---

    # STEP 1: pure demo page
    if quiz_url.endswith("/demo"):
        answer = solve_demo_page(text, html)

    # STEP 2: only the official demo-scrape page
    elif "demo-scrape?" in quiz_url:
        answer = solve_demo_scrape(text, html)

    # STEP 3: only the official demo-audio page
    elif "demo-audio" in quiz_url:
        answer = solve_demo_audio(text, html, quiz_url, official_email)


    # STEP 4: EVERYTHING ELSE → use LLM universal solver
    else:
        answer = None


    # UNIVERSAL SOLVER
    if answer is None:
        try:
            from universal_solver import solve_generic
            log("[STEP] Using Groq universal solver...")
            answer = solve_generic(html, text, quiz_url, None, email)
            log("[STEP] Universal solver result:", answer)
        except Exception as e:
            log("[STEP] Universal solver error:", e)
            answer = None

    if answer is None:
        log("[STEP] Could not compute answer")
        return None

    submit_url = submit_candidate or f"{urlparse(quiz_url).scheme}://{urlparse(quiz_url).netloc}/submit"
    submit_url = ensure_abs(quiz_url, submit_url)

    log("[STEP] Final answer:", answer)
    log("[STEP] Submit URL:", submit_url)

   # Always submit canonical STUDENT_SECRET to avoid mismatch errors
    submit_payload = {
        "email": email,
        "secret": real_secret,
        "url": quiz_url,
        "answer": answer
    }


    log("[SUBMIT] Payload:", submit_payload)

    # Try POSTing the answer
    try:
        r = requests.post(submit_url, json=submit_payload, timeout=20)
    except Exception as e:
        log("[SUBMIT ERROR] requests.post failed:", e)
        return None

    # Debug: show server response status & first 500 chars
    log("[SUBMIT] HTTP", r.status_code, "text-preview:", (r.text or "")[:500])

    # Try parsing JSON
    try:
        resp_json = r.json()
    except Exception:
        resp_json = None

    # If server returned an error
    if r.status_code >= 400:
        log("[SUBMIT] Non-200 response from submit endpoint.")
        if resp_json and isinstance(resp_json, dict):
            return resp_json.get("url")
        return None

    # If success, return next URL
    if resp_json and isinstance(resp_json, dict):
        log("[STEP] Response:", resp_json)
        return resp_json.get("url")

    log("[STEP] Submit succeeded but no JSON returned")
    return None

# ---------------------------------------------------------
# /TASK ENDPOINT
# ---------------------------------------------------------
@app.post("/task")
def run_task(req: TaskRequest):
    if req.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    log(f"[TASK] Accepted from {req.email}")

    step = 0
    next_url = req.url
    start = time.monotonic()

    while next_url and step < MAX_STEPS:
        step += 1
        if time.monotonic() - start > TOTAL_TIME_BUDGET:
            log("[TASK] Time limit exceeded")
            break

        log(f"\n==== STEP {step} ====")
        try:
        # FIXED: always use canonical secret
            next_url = process_step(next_url, req.email, STUDENT_SECRET)
        except Exception as e:
            log("[TASK] Exception:", e)
            break


    total = time.monotonic() - start
    log(f"[TASK] Finished in {int(total)}s steps={step}")

    return {"status": "done", "steps": step, "time_s": int(total)}