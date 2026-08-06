from pathlib import Path
import pandas as pd

# CHANGE THIS IF YOUR PATH IS DIFFERENT
LIBRISPEECH_ROOT = Path(
    "../dataset/LibriSpeech/test-clean"
)

records = []

transcript_files = list(
    LIBRISPEECH_ROOT.rglob("*.trans.txt")
)

print(f"Found {len(transcript_files)} transcript files")

for transcript_file in transcript_files:

    with open(transcript_file, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split(" ", 1)

            if len(parts) != 2:
                continue

            utterance_id = parts[0]
            transcript = parts[1]

            flac_path = transcript_file.parent / f"{utterance_id}.flac"

            records.append({
                "file_path": str(flac_path),
                "ground_truth": transcript
            })

df = pd.DataFrame(records)
print(df.head())
print(f"\nTotal samples found: {len(df)}")

output_path = Path("data/transcripts.csv")

df.to_csv(output_path, index=False)

print(f"Saved {len(df)} samples")
print(f"Manifest saved to {output_path}")