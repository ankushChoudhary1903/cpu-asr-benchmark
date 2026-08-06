# Low-Latency ASR Benchmarking on CPU

## Overview

This project benchmarks Whisper-family Automatic Speech Recognition (ASR) models under CPU-only inference.

The objective is to evaluate latency, throughput, and transcription accuracy tradeoffs between:

- Whisper Base (FP32)
- Whisper Base INT8 Quantized
- Distil-Whisper

The benchmark simulates a production scenario where GPU infrastructure is unavailable or cost-prohibitive, and inference must be served efficiently using CPUs.

---

## Highlights

- Benchmarked ~2,620 LibriSpeech test-clean utterances.
- Evaluated baseline, quantized, and distilled Whisper variants.
- Measured latency, throughput, Real-Time Factor (RTF), and Word Error Rate (WER).
- Implemented checkpointing and resume support for long-running benchmarks.
- Identified the optimal CPU deployment tradeoff for production inference.

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

## Benchmark Setup

Environment:

```text
Inference Backend : Faster-Whisper
Runtime           : CTranslate2
Device            : CPU
Dataset           : LibriSpeech Test-Clean
```

Models Evaluated:

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

The following metrics were collected.

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

- Lowercasing
- Punctuation removal
- Whitespace normalization

---

## Results

| Model | Avg Latency (s) | Throughput (utt/sec) | Avg RTF | WER |
|---------|---------:|---------:|---------:|---------:|
| Baseline | 1.551 | 0.645 | 0.280 | 5.76% |
| INT8 | 1.385 | 0.722 | 0.256 | 5.96% |
| Distilled | 3.328 | 0.301 | 0.618 | 5.02% |

---

## Relative Model Size

| Model | Relative Size |
|---------|---------:|
| Baseline | 1.0x |
| INT8 | ~0.25x-0.50x |
| Distilled | ~0.50x |

---

## Generated Artifacts

The benchmark produces:

- Transcription outputs for each model
- Aggregate evaluation metrics
- WER reports
- Latency comparison plots
- Accuracy-vs-latency tradeoff visualizations

---

## Key Findings

### INT8 Quantization

Compared to the FP32 baseline:

- ~10.7% lower latency
- ~12% higher throughput
- Only ~0.20% WER degradation

This provided the best latency-to-accuracy tradeoff.

### Distillation

Distil-Whisper achieved the best transcription accuracy:

```text
WER = 5.02%
```

However:

- Inference latency increased significantly
- Throughput decreased substantially

On the evaluated hardware, distillation did not provide the expected deployment benefits compared to quantized inference.

---

## Recommendation

For CPU-only production deployment:

```text
Whisper Base INT8
```

is the preferred option because it provides:

- Lower latency
- Higher throughput
- Nearly identical transcription accuracy

while requiring fewer computational resources.

Although Distil-Whisper achieved the lowest WER (5.02%), INT8 quantization delivered significantly lower latency (1.385s vs. 3.328s) while maintaining comparable accuracy. Therefore, INT8 Whisper Base offers the best latency-accuracy tradeoff for CPU-only deployment.

---

## Project Structure

```text
cpu-asr-benchmark/
│
├── data/
│   └── transcripts.csv
│
├── evaluation/
│   ├── aggregate_metrics.py
│   ├── plot_results.py
│   └── wer_evaluator.py
│
├── results/
│   ├── metrics.csv
│   ├── wer_results.csv
│   ├── final_summary.md
│   └── plots/
│
├── runners/
│   ├── baseline_runner.py
│   ├── quantized_runner.py
│   └── distilled_runner.py
│
├── utils/
│   ├── build_manifest.py
│   └── data_loader.py
│
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies

- Python
- Faster-Whisper
- CTranslate2
- JiWER
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Reproducibility

### 1. Download Dataset

Download LibriSpeech Test-Clean and place it under:

```text
dataset/LibriSpeech/test-clean/
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Benchmarks

```bash
python runners/baseline_runner.py
python runners/quantized_runner.py
python runners/distilled_runner.py
```

### 4. Evaluate Results

```bash
python evaluation/wer_evaluator.py
python evaluation/aggregate_metrics.py
python evaluation/plot_results.py
```

---

## Future Improvements

- Multi-thread CPU benchmarking
- Memory consumption analysis
- Multilingual evaluation
- Streaming ASR benchmarking
- Real-time deployment testing
- CPU thread scaling analysis
- Dockerized deployment benchmarking

---