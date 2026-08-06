import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import pandas as pd
from faster_whisper import WhisperModel

from utils.data_loader import load_manifest

from config import (
    TRANSCRIPT_CSV,
    DISTIL_MODEL_NAME,
    DEVICE,
    BASE_COMPUTE_TYPE,
    DISTIL_OUTPUT,
    CHECKPOINT_INTERVAL
)

print("Loading Distil-Whisper model...")

model = WhisperModel(
    DISTIL_MODEL_NAME,
    device=DEVICE,
    compute_type=BASE_COMPUTE_TYPE
)

print("Model loaded.")

df = load_manifest(TRANSCRIPT_CSV)

output_file = DISTIL_OUTPUT

results = []

if Path(output_file).exists():

    existing_df = pd.read_csv(output_file)

    processed_files = set(existing_df["file_path"])

    results = existing_df.to_dict("records")

    print(
        f"Resuming from {len(processed_files)} completed files"
    )

else:

    processed_files = set()

for idx, row in df.iterrows():

    audio_path = row["file_path"]

    if audio_path in processed_files:
        continue

    start = time.perf_counter()

    segments, info = model.transcribe(
        audio_path,
        language="en"
    )

    prediction = " ".join(
        segment.text.strip()
        for segment in segments
    )

    latency = time.perf_counter() - start

    duration = info.duration

    rtf = latency / max(duration, 1e-6)

    results.append({
        "file_path": audio_path,
        "ground_truth": row["ground_truth"],
        "prediction": prediction,
        "latency_sec": latency,
        "audio_duration_sec": duration,
        "rtf": rtf
    })

    if idx % CHECKPOINT_INTERVAL == 0:

        pd.DataFrame(results).to_csv(
            output_file,
            index=False
        )

        print(
            f"Checkpoint saved ({len(results)} rows)"
        )

    if idx % 50 == 0:
        print(
            f"Processed {idx}/{len(df)}"
        )

pd.DataFrame(results).to_csv(
    output_file,
    index=False
)

print(f"Saved results to {output_file}")