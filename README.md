# 8x8 Class Quality Auditor

Upload a class recording and get a **Poor / Average / Good / Excellent** rating with
timestamped evidence, a rubric check and concrete recommendations.

## How it works

1. **Audio** (`signals.py`): decoded to 16 kHz mono and measured on 50 ms RMS windows with an
   adaptive threshold: time speaking, dead air (silences > 5 s), longest silence.
2. **Video** (`signals.py`): one frame sampled every ~5 s by seeking: camera off/black, face
   detected, brightness, sharpness, how often the picture changes.
3. **Transcript**: uploaded `.vtt`, or transcribed **locally** with faster-whisper, so audio never
   leaves the server. Extracts greeting in the first 2 min, homework near the end, questions per
   10 min, and personal-information mentions.
4. **LLM judgement** (`assess.py`): Gemini gets the full timestamped transcript, the signals and
   rule-based flags, and returns structured JSON (rating, confidence, evidence with timestamps,
   rubric check, recommendations, and which signals it discounted). Temperature 0.

The computed signals are treated as **hints, not ground truth**. The transcript is the primary
evidence, and the model is told to discount signals that are likely misleading.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your key
streamlit run app.py
```

## Study the dataset (calibration)

```bash
python calibrate.py samples            # samples/<excellent|good|average|weak>/*.mp4
python calibrate.py samples --transcribe base
```

Prints the mean of every signal per tier and saves `signals.csv`. Keep the signals that
separate tiers, drop the rest, and update `THRESHOLDS` (`signals.py`) and the rubric (`assess.py`).

## Confidentiality

* Recordings are processed in a temporary directory and never stored or committed
  (`.gitignore` blocks media, transcripts and `samples/`).
* Only a **name-redacted transcript and computed numbers** go to the LLM, never video or audio, and
  only after the consent box is ticked. Without it the app shows local signals only.
* Set `APP_PASSWORD` in the hosting secrets to gate the public app.
* Confirm with the organiser that the chosen LLM service is permitted.

## Deploy (Streamlit Community Cloud)

Push to GitHub, deploy `app.py`, and paste the contents of `secrets.toml.example` (with real
values) into **Advanced settings > Secrets**.

## Limitations

* Thresholds are heuristics until calibrated on the sample set.
* Face detection can miss small video tiles; silence can be children working quietly.
* No speaker labels, so teacher vs. student talk is not separated.
* Local Whisper is slow on free hosting; upload a `.vtt` for long classes.
* Very long transcripts are truncated (head + tail) before the LLM call.
