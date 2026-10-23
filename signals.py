"""Signal extraction for class recordings.

Pure helpers with no Streamlit dependency, so they are shared by the app
(app.py) and the dataset study script (calibrate.py).

Design notes
------------
* Audio is decoded to 16 kHz mono and processed in small chunks, so a long
  class never needs gigabytes of RAM.
* Silence is measured on 50 ms RMS windows with an adaptive threshold, not on
  raw samples (raw samples cross zero constantly and look "silent").
* Video is sampled by seeking (one frame every few seconds), not by decoding
  every frame.
* Every number here is a *hint* for the LLM and for the reviewer. None of the
  thresholds are validated until you run calibrate.py on the sample dataset.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass

import cv2
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Heuristic thresholds. NOT validated: tune with calibrate.py.
THRESHOLDS = {
    "dark_ratio_camera_off": 0.5,
    "face_ratio_low": 0.3,
    "brightness_dim": 70,
    "brightness_over": 200,
    "sharpness_low": 40,
    "long_silence_s": 5.0,
    "long_silence_ratio": 0.25,
    "min_questions_per_10min": 3,
}


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
def extract_audio(video_path: str, wav_path: str) -> bool:
    """Decode the audio track to 16 kHz mono WAV. False if there is no audio."""
    cmd = [FFMPEG, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", wav_path]
    result = subprocess.run(cmd, capture_output=True)
    return (result.returncode == 0 and os.path.exists(wav_path)
            and os.path.getsize(wav_path) > 44)


def _runs(mask: np.ndarray):
    """(start_index, length) for each run of consecutive True values."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends - starts))


def audio_metrics(wav_path: str, win_ms: int = 50) -> dict:
    chunks = []
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        win = int(sr * win_ms / 1000)
        while True:
            raw = w.readframes(win * 400)  # ~20 s per chunk
            if not raw:
                break
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            k = len(x) // win
            if k == 0:
                break
            x = x[: k * win].reshape(k, win)
            rms = np.sqrt((x ** 2).mean(axis=1))
            chunks.append(20 * np.log10(rms + 1e-9))
    if not chunks:
        return {"has_audio": False}

    db = np.concatenate(chunks)
    step = win_ms / 1000.0
    duration = len(db) * step
    p95 = float(np.percentile(db, 95))
    near_silent = p95 < -55
    threshold = p95 - 22  # speech is >= ~22 dB above the quiet floor
    voiced = np.zeros_like(db, dtype=bool) if near_silent else db > threshold

    min_len = int(THRESHOLDS["long_silence_s"] / step)
    long_runs = [(s, n) for s, n in _runs(~voiced) if n >= min_len]
    long_silence_windows = sum(n for _, n in long_runs)

    per_min_len = int(60 / step)
    per_min = [float(voiced[i:i + per_min_len].mean())
               for i in range(0, len(voiced), per_min_len)]

    return {
        "has_audio": True,
        "near_silent": near_silent,
        "duration_s": round(duration, 1),
        "speech_ratio": round(float(voiced.mean()), 3),
        "long_silence_ratio": round(long_silence_windows / len(db), 3),
        "n_long_silences": len(long_runs),
        "longest_silence_s": round(max((n for _, n in long_runs), default=0) * step, 1),
        # weak proxy for vocal expressiveness: spread of loudness while speaking
        "loudness_spread_db": round(float(db[voiced].std()), 2) if voiced.any() else 0.0,
        "speech_per_min": per_min,
    }


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------
def video_metrics(video_path: str, every_s: float = 5.0, max_samples: int = 240) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if fps <= 0 or n_frames <= 0:
        cap.release()
        return {"readable": False}
    duration = n_frames / fps
    times = np.arange(0, duration, every_s)
    if len(times) > max_samples:
        times = np.linspace(0, max(duration - 1, 0), max_samples)

    # Face detection is optional: if this OpenCV build lacks it (e.g. two OpenCV
    # packages installed side by side), skip it rather than failing the whole run.
    try:
        cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        if cascade.empty():
            cascade = None
    except Exception:  # noqa: BLE001
        cascade = None

    brightness, contrast, sharpness, dark, faces_seen, small = [], [], [], [], [], []
    width = height = 0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        scale = 320.0 / width
        frame = cv2.resize(frame, (320, max(int(height * scale), 1)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b, c = float(gray.mean()), float(gray.std())
        brightness.append(b)
        contrast.append(c)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        dark.append(b < 25 or c < 10)          # black / flat frame => camera off
        if cascade is not None:
            found = cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
            faces_seen.append(len(found) > 0)
        small.append(cv2.GaussianBlur(cv2.resize(gray, (160, 90)), (5, 5), 0).astype(np.float32))
    cap.release()

    if not brightness:
        return {"readable": False}

    diffs = [float(np.abs(a - b).mean()) for a, b in zip(small[:-1], small[1:])]
    return {
        "readable": True,
        "duration_s": round(duration, 1),
        "resolution": f"{width}x{height}",
        "n_samples": len(brightness),
        "mean_brightness": round(float(np.mean(brightness)), 1),
        "mean_contrast": round(float(np.mean(contrast)), 1),
        "mean_sharpness": round(float(np.mean(sharpness)), 1),
        "dark_ratio": round(float(np.mean(dark)), 3),
        "face_ratio": round(float(np.mean(faces_seen)), 3) if faces_seen else None,
        # how often the picture changes between samples (slides, board, movement)
        "change_ratio": round(float(np.mean([d > 4 for d in diffs])), 3) if diffs else 0.0,
        # near-identical consecutive samples (frozen video / camera off / static slide)
        "frozen_ratio": round(float(np.mean([d < 0.8 for d in diffs])), 3) if diffs else 0.0,
    }


# --------------------------------------------------------------------------
# Transcript
# --------------------------------------------------------------------------
@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str = ""


SPEAKER_RE = re.compile(r"^([^:]{1,40}):\s+(.*)$")


def parse_vtt(path: str) -> list[Cue]:
    import webvtt
    raw = []
    for c in webvtt.read(path):
        text = re.sub(r"\s+", " ", c.text).strip()
        if text:
            raw.append((c.start_in_seconds, c.end_in_seconds, text))
    matches = [SPEAKER_RE.match(t) for _, _, t in raw]
    has_speakers = bool(raw) and sum(m is not None for m in matches) / len(raw) >= 0.5
    cues, last = [], ""
    for (start, end, text), m in zip(raw, matches):
        speaker = ""
        if has_speakers:
            if m:
                last, text = m.group(1).strip(), m.group(2).strip()
            speaker = last
        if text:
            cues.append(Cue(start, end, text, speaker))
    return cues


_NAME_STOP = {"user", "iphone", "ipad", "phone", "admin", "guest", "host", "zoom"}


def pseudonymize_speakers(cues: list[Cue]) -> list[Cue]:
    """Replace speaker labels with TEACHER / STUDENT_n (teacher = most words) and
    scrub the speakers' own names from the text, so no real names leave the server."""
    if not any(c.speaker for c in cues):
        return cues
    words = {}
    for c in cues:
        words[c.speaker] = words.get(c.speaker, 0) + len(c.text.split())
    teacher = max(words, key=words.get)
    labels, n = {}, 0
    for c in cues:
        if c.speaker not in labels:
            if c.speaker == teacher:
                labels[c.speaker] = "TEACHER"
            else:
                n += 1
                labels[c.speaker] = f"STUDENT_{n}"
    pats, seen = [], set()
    for name, label in labels.items():
        for tok in re.findall(r"[A-Za-z]{3,}", name):
            if tok.lower() in _NAME_STOP or tok.lower() in seen:
                continue
            seen.add(tok.lower())
            pats.append((re.compile(r"\b%s\b" % re.escape(tok), re.I), f"[{label}]"))
    out = []
    for c in cues:
        text = c.text
        for pat, rep in pats:
            text = pat.sub(rep, text)
        out.append(Cue(c.start, c.end, text, labels[c.speaker]))
    return out


def transcribe_local(wav_path: str, model_size: str = "base", progress=None,
                     chunk_s: int = 600) -> list[Cue]:
    """Transcribe on this machine (audio never leaves the server).

    The WAV is read here and passed to Whisper as an array in 10-minute chunks, which
    avoids Whisper's own PyAV decoder (a version mismatch there caused errors) and keeps
    memory low for long classes.
    """
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    cues, language, done = [], None, 0
    with wave.open(wav_path, "rb") as w:
        sr, total = w.getframerate(), w.getnframes()
        while done < total:
            raw = w.readframes(sr * chunk_s)
            if not raw:
                break
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < sr:  # ignore a sub-second trailing fragment
                break
            offset = done / sr
            segments, info = model.transcribe(audio, language=language,
                                              vad_filter=True, beam_size=1)
            language = language or info.language  # detect once, reuse for later chunks
            for seg in segments:
                text = seg.text.strip()
                if text:
                    cues.append(Cue(offset + float(seg.start), offset + float(seg.end), text))
            done += len(audio)
            if progress:
                progress(min(done / total, 1.0))
    return cues


def redact(cues: list[Cue], names: list[str]) -> list[Cue]:
    """Replace listed names with [STUDENT_n] so name-calling can still be counted."""
    names = [n.strip() for n in names if n.strip()]
    if not names:
        return cues
    pats = [(re.compile(r"\b%s\b" % re.escape(n), re.I), f"[STUDENT_{i + 1}]")
            for i, n in enumerate(names)]
    out = []
    for c in cues:
        text = c.text
        for pat, rep in pats:
            text = pat.sub(rep, text)
        out.append(Cue(c.start, c.end, text, c.speaker))
    return out


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


GREETING_RE = re.compile(
    r"\b(hello|hi|hey|good (morning|afternoon|evening)|welcome|how are you|how's everyone)\b", re.I)
HOMEWORK_RE = re.compile(
    r"\b(homework|home work|assignment|worksheet|practi[cs]e (this )?at home|"
    r"for next (class|time|session)|before (the )?next (class|session))\b", re.I)
QUESTION_START_RE = re.compile(
    r"^(what|why|how|who|where|when|which|can you|could you|do you|did you|are you|"
    r"is it|have you|will you|would you)\b", re.I)
PRAISE_RE = re.compile(
    r"\b(very good|good job|well done|perfect|excellent|great|nice|awesome|wonderful)\b", re.I)
PRIVACY_RE = re.compile(
    r"\b(phone number|mobile number|your number|home address|your address|where do you live|"
    r"e-?mail|whatsapp|password|last name|which school)\b", re.I)


def transcript_signals(cues: list[Cue], duration_s: float) -> dict:
    if not cues:
        return {"available": False}
    duration_s = duration_s or cues[-1].end
    full = " ".join(c.text for c in cues)
    words = len(full.split())

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full) if s.strip()]
    q_like = sum(1 for s in sentences if s.endswith("?") or QUESTION_START_RE.match(s))

    early = " ".join(c.text for c in cues if c.start < 120)
    late = " ".join(c.text for c in cues if c.start > 0.8 * duration_s)

    privacy_mentions = []
    for i, c in enumerate(cues):
        if PRIVACY_RE.search(c.text) and len(privacy_mentions) < 6:
            ctx = " ".join(x.text for x in cues[max(0, i - 1): i + 2])
            privacy_mentions.append({"time": fmt_time(c.start), "context": ctx})

    minutes = max(duration_s / 60.0, 1e-9)
    speaker_stats = {}
    if any(c.speaker for c in cues):
        t_cues = [c for c in cues if c.speaker == "TEACHER"]
        s_cues = [c for c in cues if c.speaker and c.speaker != "TEACHER"]
        tw = sum(len(c.text.split()) for c in t_cues)
        sw = sum(len(c.text.split()) for c in s_cues)
        t_text = " ".join(c.text for c in t_cues)
        called = re.findall(r"\[STUDENT_\d+\]", t_text)
        speaker_stats = {
            "teacher_word_share": round(tw / max(tw + sw, 1), 3),
            "students_who_spoke": len({c.speaker for c in s_cues}),
            "teacher_questions_per_10min": round(t_text.count("?") / minutes * 10, 1),
            # undercount likely: speech-to-text mangles names
            "teacher_name_calls": len(called),
            "distinct_students_called_by_name": len(set(called)),
            "teacher_praise_per_10min": round(len(PRAISE_RE.findall(t_text)) / minutes * 10, 1),
        }
    return {
        **speaker_stats,
        "available": True,
        "words": words,
        "words_per_min": round(words / minutes, 1),
        "question_marks": full.count("?"),
        "question_like_sentences": q_like,
        "questions_per_10min": round(q_like / minutes * 10, 1),
        "greeting_in_first_2min": bool(GREETING_RE.search(early)),
        "homework_mentioned_anywhere": bool(HOMEWORK_RE.search(full)),
        "homework_mentioned_in_last_20pct": bool(HOMEWORK_RE.search(late)),
        "privacy_mentions": privacy_mentions,
    }


# --------------------------------------------------------------------------
# Rule-based flags (shown to the reviewer and passed to the LLM as hints)
# --------------------------------------------------------------------------
def heuristic_flags(audio: dict, video: dict, text: dict) -> list[dict]:
    T = THRESHOLDS
    flags = []

    def add(kind, msg):
        flags.append({"kind": kind, "msg": msg})

    if not audio.get("has_audio") or audio.get("near_silent"):
        add("risk", "No usable audio track detected.")
    else:
        if audio["long_silence_ratio"] > T["long_silence_ratio"]:
            add("risk", f"{audio['long_silence_ratio']:.0%} of the class is in silences "
                        f">{T['long_silence_s']:.0f}s (longest {audio['longest_silence_s']}s).")

    if video.get("readable"):
        if video["dark_ratio"] > T["dark_ratio_camera_off"]:
            add("risk", f"Camera looks off or black in {video['dark_ratio']:.0%} of sampled frames.")
        elif video["face_ratio"] is not None and video["face_ratio"] < T["face_ratio_low"]:
            add("info", f"A face was detected in only {video['face_ratio']:.0%} of frames "
                        "(could be camera angle, screen-share only, or a tiny video tile). Verify.")
        if video["dark_ratio"] <= T["dark_ratio_camera_off"]:  # skip if camera is simply off
            if video["mean_brightness"] < T["brightness_dim"]:
                add("risk", f"Dim lighting (mean brightness {video['mean_brightness']}).")
            elif video["mean_brightness"] > T["brightness_over"]:
                add("risk", f"Over-exposed picture (mean brightness {video['mean_brightness']}).")
            if video["mean_sharpness"] < T["sharpness_low"]:
                add("risk", f"Low picture sharpness ({video['mean_sharpness']}); "
                            "possible blur or poor clarity.")

    if text.get("available"):
        if not text["greeting_in_first_2min"]:
            add("risk", "No greeting or rapport-building detected in the first 2 minutes.")
        if not text["homework_mentioned_anywhere"]:
            add("info", "Homework or follow-up work is never mentioned (weak signal: not every strong class sets it).")
        elif not text["homework_mentioned_in_last_20pct"]:
            add("info", "Homework is mentioned, but not near the end of the class.")
        qpm = text.get("teacher_questions_per_10min", text["questions_per_10min"])
        if qpm < T["min_questions_per_10min"]:
            add("risk", f"Few questions to students ({qpm} per 10 min).")
        if text.get("teacher_word_share", 0) > 0.8:
            add("info", f"Teacher speaks {text['teacher_word_share']:.0%} of all words; "
                        "students get little airtime.")
        if text["privacy_mentions"]:
            add("info", f"{len(text['privacy_mentions'])} mention(s) of personal information. "
                        "Needs context to judge whether the teacher enforced or broke the rule.")
    else:
        add("info", "No transcript available: the assessment relies on audio/video signals only.")
    return flags


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def analyze(video_path: str, vtt_path: str | None = None, whisper_model: str | None = None,
            names: list[str] | None = None, progress=None) -> dict:
    def step(frac, msg):
        if progress:
            progress(frac, msg)

    cues, source = [], "none"
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        step(0.05, "Extracting audio...")
        has_wav = extract_audio(video_path, wav)
        step(0.15, "Measuring speech and silence...")
        audio = audio_metrics(wav) if has_wav else {"has_audio": False}
        step(0.30, "Sampling video frames...")
        video = video_metrics(video_path)
        if vtt_path:
            step(0.50, "Reading transcript...")
            cues, source = parse_vtt(vtt_path), "uploaded .vtt"
        elif whisper_model and has_wav:
            cues = transcribe_local(
                wav, whisper_model,
                progress=lambda f: step(0.40 + 0.50 * f, "Transcribing on this server..."))
            source = f"local Whisper ({whisper_model})"

    duration = (audio.get("duration_s") or video.get("duration_s")
                or (cues[-1].end if cues else 0))
    cues = redact(pseudonymize_speakers(cues), names or [])
    text = transcript_signals(cues, duration)
    return {
        "duration_s": duration,
        "audio": audio,
        "video": video,
        "text": text,
        "flags": heuristic_flags(audio, video, text),
        "cues": cues,
        "transcript_source": source,
    }
