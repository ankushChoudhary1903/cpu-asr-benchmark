import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import pandas as pd
from faster_whisper import WhisperModel

from utils.data_loader import load_manifest
from config import (
    TRANSCRIPT_CSV,
    MODEL_NAME,
    DEVICE,
    COMPUTE_TYPE,
    BASELINE_OUTPUT
)

print("Loading model...")

model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

print("Model loaded.")

df = load_manifest(TRANSCRIPT_CSV)

results = []

for idx, row in df.iterrows():

    audio_path = row["file_path"]

    start = time.perf_counter()

    segments, info = model.transcribe(audio_path)

    prediction = " ".join(
        segment.text.strip()
        for segment in segments
    )

    latency = time.perf_counter() - start

    duration = info.duration

    rtf = latency / duration

    results.append({
        "file_path": audio_path,
        "ground_truth": row["ground_truth"],
        "prediction": prediction,
        "latency_sec": latency,
        "audio_duration_sec": duration,
        "rtf": rtf
    })

    if idx % 50 == 0:
        print(f"Processed {idx}/{len(df)}")

results_df = pd.DataFrame(results)

results_df.to_csv(
    BASELINE_OUTPUT,
    index=False
)

print(f"Saved results to {BASELINE_OUTPUT}")