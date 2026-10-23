# 8x8 Class Quality Auditor

Upload a class recording and get a **Poor / Average / Good / Excellent** rating with
timestamped evidence, a rubric check and concrete recommendations. Built for the Humaniti AI
"Can you teach AI to recognise a great class?" challenge.

## How it works

1. **Audio** (`signals.py`): the audio track is decoded to 16 kHz mono and measured in
   50 ms RMS windows with an adaptive threshold. Outputs: time speaking, dead air (silences
   over 5 s) and the longest silence. It is processed in chunks, so long classes do not run
   out of memory.
2. **Video** (`signals.py`): one frame is sampled every ~5 s by seeking. Outputs: camera
   off/black, brightness, sharpness, how often the picture changes, and (when OpenCV's face
   detector is available) how often a face is visible. Face detection is optional; if it is
   unavailable the run continues without it.
3. **Transcript**: an uploaded `.vtt`, or, if none is uploaded, local transcription with
   faster-whisper (the audio never leaves the server). Whisper receives the audio in
   10-minute chunks. From the transcript the app derives: greeting in the first 2 minutes,
   homework mentions, questions per 10 minutes, praise rate, and personal-information
   mentions (with context). When the `.vtt` has speaker labels, it also derives the
   teacher's share of words, how many students spoke, and name-calling.
4. **LLM judgement** (`assess.py`): an OpenAI-compatible chat API (Groq by default) receives
   the full timestamped transcript, the computed signals and rule-based flags, and returns
   structured JSON: rating, confidence, summary, evidence with timestamps, a rubric check,
   recommendations, and which signals it discounted. Temperature 0. If the provider rejects
   the request as too large or rate-limited, the app retries with a shorter transcript.

The computed signals are **hints, not ground truth**. The transcript is the primary evidence,
and the model is told to discount signals that are likely to mislead.

## Rubric (starting hypotheses)

- **Excellent**: children are actively involved throughout; frequent questions, every child
  gets repeated turns, praise and encouragement, warm rapport from the start, and the lesson
  keeps moving through technical problems. Homework is a bonus, not a requirement.
- **Good**: solid structure and participation, but some children get little airtime, or
  warmth, energy or clarity is uneven.
- **Average**: basic interaction (mostly teacher talk or procedural questions), little
  rapport, or audio/video problems that clearly hurt the lesson.
- **Poor**: no introduction, little or no student interaction, long unexplained dead air, or
  the camera is off / audio unusable.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your key
streamlit run app.py
```

Secrets / environment variables:

| Name | Purpose |
|------|---------|
| `GROQ_API_KEY` | API key for the LLM provider (or `LLM_API_KEY`) |
| `LLM_MODEL` | Model ID, for example `openai/gpt-oss-120b` |
| `LLM_BASE_URL` | Defaults to `https://api.groq.com/openai/v1`; any OpenAI-compatible endpoint works |
| `APP_PASSWORD` | Optional; gates the hosted app |

Use `opencv-python-headless` only. Installing it next to `opencv-python` breaks `cv2`.

## Study the dataset (calibration)

```bash
python calibrate.py samples                    # samples/<excellent|good|average|weak>/*.mp4
python calibrate.py samples --transcribe base  # also transcribe recordings that have no .vtt
```

Put each `.vtt` next to its `.mp4` with the same name. The script prints the mean of every
signal per tier, saves `signals.csv`, and writes `reference_profiles.json` (averages only, no
recordings or transcripts). If that file is present in the repo, `assess.py` shows it to the
model so its rating is anchored to the labelled samples. Use the table to keep the signals
that separate tiers, drop the rest, and update `THRESHOLDS` (`signals.py`) and the rubric
(`assess.py`).

## Confidentiality

- Recordings are processed in a temporary directory and are never stored or committed
  (`.gitignore` blocks media, transcripts and `samples/`).
- The video and audio are never sent to the LLM. Only the transcript and computed numbers
  are, and only after the consent box is ticked. Without it the app shows local signals only.
- **Names:** if the `.vtt` has speaker labels, speakers are replaced with `TEACHER` /
  `STUDENT_n` and their names are scrubbed from the text. Speech-to-text spells names
  inconsistently, so some spoken names can remain. List the names and likely misspellings in
  the redaction box to replace them. With a video-only upload (local Whisper) there are no
  speaker labels, so only the names you list are redacted.
- Set `APP_PASSWORD` in the hosting secrets to gate the public app.
- The LLM provider is a third-party service. Confirm with the organiser that it is permitted
  and read its data-use terms before using real recordings.

## Deploy (Streamlit Community Cloud)

Push to GitHub, deploy `app.py`, and paste the contents of `secrets.toml.example` (with real
values) into **Advanced settings > Secrets**. `requirements.txt` must list
`opencv-python-headless` and not `opencv-python`. If a dependency error persists after a
change, delete and redeploy the app for a clean install.

## Findings from the sample set

- **Misleading signals:** sharpness depends on on-screen content (it was very high on a
  class that was visually ordinary), a high "frozen" ratio also occurs when children sit
  still, and silences of several seconds are normal while children solve puzzles. These are
  passed to the model as hints with that context, not as verdicts.
- **Homework is not a reliable marker of quality:** a sample labelled Excellent never sets
  homework, so the rubric treats it as a bonus.
- **Known gap:** the model under-rated that Excellent sample (Average). It weighed small
  technical glitches and quiet stretches too heavily and misread a countdown game as unclear
  instruction. In video-only mode there are no speaker labels, so the app cannot see how
  evenly the teacher shared turns among children.

## Limitations

- Thresholds and the rubric are heuristics until calibrated on more labelled samples.
- Face detection can miss small video tiles, and silence can be children working quietly.
- Without speaker labels, teacher and student talk are not separated.
- Local Whisper is slow on free hosting; upload a `.vtt` for long classes. Whisper also
  garbles children's names.
- Very long transcripts are shortened (start and end kept) before the LLM call.
- Ratings come from an LLM and can vary between runs and models.
