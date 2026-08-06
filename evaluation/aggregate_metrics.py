import pandas as pd

files = {
    "Baseline": "results/predictions/baseline.csv",
    "INT8": "results/predictions/int8.csv",
    "Distilled": "results/predictions/distilled.csv",
}

rows = []

for model_name, file_path in files.items():

    df = pd.read_csv(file_path)

    avg_latency = df["latency_sec"].mean()

    avg_rtf = df["rtf"].mean()

    throughput = 1 / avg_latency

    rows.append({
        "model": model_name,
        "avg_latency_sec": avg_latency,
        "throughput_utt_per_sec": throughput,
        "avg_rtf": avg_rtf
    })

metrics_df = pd.DataFrame(rows)

wer_df = pd.read_csv(
    "results/wer_results.csv"
)

metrics_df["wer"] = wer_df["wer"]

metrics_df.to_csv(
    "results/metrics.csv",
    index=False
)

print(metrics_df)