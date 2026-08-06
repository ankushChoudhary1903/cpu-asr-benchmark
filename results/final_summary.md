# CPU ASR Benchmark Results

Dataset:
LibriSpeech Test-Clean (~2620 utterances)

Models Evaluated:
1. Whisper Base (FP32)
2. Whisper Base INT8
3. Distil-Whisper

Results

| Model | Latency (s) | Throughput | WER |
|---------|---------|---------|---------|
| Baseline | 1.551 | 0.645 | 5.76% |
| INT8 | 1.385 | 0.722 | 5.96% |
| Distilled | 3.328 | 0.301 | 5.02% |

Key Findings

- INT8 reduced latency by ~10.7%.
- INT8 increased throughput by ~12%.
- Accuracy degradation was only ~0.20% WER.
- Distilled Whisper achieved the best WER but had the worst latency.
- For CPU-only production inference, INT8 provides the best latency-accuracy tradeoff.