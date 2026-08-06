# Low-Latency ASR Benchmarking on CPU

## Overview

This project benchmarks Whisper-family Automatic Speech Recognition (ASR) models under CPU-only inference.

The objective is to evaluate latency, throughput, and transcription accuracy tradeoffs between:

- Whisper Base (FP32)
- Whisper Base INT8 Quantized
- Distil-Whisper

The benchmark simulates a production scenario where GPU infrastructure is unavailable or cost-prohibitive, and inference must be served efficiently using CPUs.

---

## Objective

Identify the best accuracy-per-latency tradeoff for production-scale CPU inference.

Questions investigated:

- How much latency improvement does INT8 quantization provide?
- How much transcription accuracy is sacrificed?
- Does model distillation provide a better tradeoff than quantization?
- Which model should be deployed under CPU constraints?

---

## Dataset

Dataset:

```text
LibriSpeech Test-Clean
```

Characteristics:

- Approximately 2,620 audio utterances
- Multiple speakers
- Human-verified transcripts
- Fixed evaluation set used across all experiments

This dataset remained unchanged across all benchmark runs to ensure fair comparison.

---

## Models Evaluated

### Baseline

```text
Whisper Base
FP32
CPU Only
```

### Quantized

```text
Whisper Base
INT8 Quantized
CPU Only
```

### Distilled

```text
Distil-Whisper Small
CPU Only
```

---

## Metrics

The following metrics were collected:

### Latency

Average inference time per utterance.

### Throughput

Utterances processed per second.

### Real-Time Factor (RTF)

```text
Inference Time / Audio Duration
```

Lower values indicate faster inference.

### Word Error Rate (WER)

Calculated using JiWER after:

- lowercasing
- punctuation removal
- whitespace normalization

---

## Results

| Model     | Avg Latency (s) | Throughput (utt/sec) | Avg RTF |   WER |
| --------- | --------------: | -------------------: | ------: | ----: |
| Baseline  |           1.551 |                0.645 |   0.280 | 5.76% |
| INT8      |           1.385 |                0.722 |   0.256 | 5.96% |
| Distilled |           3.328 |                0.301 |   0.618 | 5.02% |

---

## Key Findings

### INT8 Quantization

Compared to the FP32 baseline:

- ~10.7% lower latency
- ~12% higher throughput
- Only 0.20% WER degradation

This provided the best latency-to-accuracy tradeoff.

### Distillation

Distil-Whisper achieved the best WER:

```text
5.02%
```

However:

- inference latency increased significantly
- throughput decreased substantially

On the evaluated hardware, distillation did not provide the expected deployment benefits compared to quantized inference.

---

## Recommendation

For CPU-only production deployment:

```text
Whisper Base INT8
```

is the preferred option because it provides:

- lower latency
- higher throughput
- nearly identical transcription accuracy

while requiring fewer computational resources.

---

## Project Structure

```text
cpu-asr-benchmark/
│
├── data/
├── evaluation/
├── results/
│   ├── predictions/
│   └── plots/
├── runners/
├── utils/
│
├── config.py
├── requirements.txt
└── README.md
```

---

## Technologies

- Python
- Faster-Whisper
- CTranslate2
- JiWER
- Pandas
- Matplotlib

---

## Reproducibility

1. Download LibriSpeech Test-Clean
2. Place dataset under:

dataset/LibriSpeech/test-clean/

3. Install dependencies:

pip install -r requirements.txt

4. Run:

python runners/baseline_runner.py
python runners/quantized_runner.py
python runners/distilled_runner.py

5. Evaluate:

python evaluation/wer_evaluator.py
python evaluation/aggregate_metrics.py
python evaluation/plot_results.py

---

## Future Improvements

- Multi-thread CPU benchmarking
- Memory consumption analysis
- Multilingual evaluation
- Streaming ASR benchmarking
- Real-time deployment testing
