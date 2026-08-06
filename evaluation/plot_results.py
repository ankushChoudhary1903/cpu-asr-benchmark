import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/metrics.csv"
)
df["wer_percent"] = df["wer"] * 100

# Latency

plt.figure(figsize=(8,5))

plt.bar(
    df["model"],
    df["avg_latency_sec"]
)

plt.title("Average Latency")
plt.ylabel("Seconds")

plt.tight_layout()

plt.savefig(
    "results/plots/latency_comparison.png"
)

plt.close()

# WER

plt.figure(figsize=(8,5))

plt.bar(
    df["model"],
    df["wer_percent"]
)

plt.title("WER Comparison")
plt.ylabel("WER (%)")

plt.tight_layout()

plt.savefig(
    "results/plots/wer_comparison.png"
)

plt.close()

# Tradeoff

plt.figure(figsize=(8,5))

plt.scatter(
    df["avg_latency_sec"],
    df["wer_percent"]
)

for _, row in df.iterrows():

    plt.annotate(
        row["model"],
        (
            row["avg_latency_sec"],
            row["wer_percent"]
        )
    )

plt.xlabel("Latency (sec)")
plt.ylabel("WER (%)")

plt.title("Latency vs Accuracy")

plt.tight_layout()

plt.savefig(
    "results/plots/tradeoff_curve.png"
)