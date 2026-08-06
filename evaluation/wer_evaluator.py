import pandas as pd
from jiwer import wer

FILES = {
    "Baseline": "results/predictions/baseline.csv",
    "INT8": "results/predictions/int8.csv",
    "Distilled": "results/predictions/distilled.csv",
}


def normalize(text):
    text = str(text).lower()
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace(";", "")
    text = text.replace(":", "")
    text = " ".join(text.split())
    return text


results = []

for model_name, file_path in FILES.items():

    df = pd.read_csv(file_path)

    references = [
        normalize(x)
        for x in df["ground_truth"]
    ]

    predictions = [
        normalize(x)
        for x in df["prediction"]
    ]

    score = wer(
        references,
        predictions
    )

    results.append({
        "model": model_name,
        "wer": score
    })

wer_df = pd.DataFrame(results)

wer_df.to_csv(
    "results/wer_results.csv",
    index=False
)

print("\nWER Results")
print(wer_df)