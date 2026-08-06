# Dataset
TRANSCRIPT_CSV = "data/transcripts.csv"

# Models
BASE_MODEL_NAME = "base"
DISTIL_MODEL_NAME = "Systran/faster-distil-whisper-small.en"

# Device
DEVICE = "cpu"

# Compute Types
BASE_COMPUTE_TYPE = "float32"
INT8_COMPUTE_TYPE = "int8"

# Output Files
BASELINE_OUTPUT = "results/predictions/baseline.csv"
INT8_OUTPUT = "results/predictions/int8.csv"
DISTIL_OUTPUT = "results/predictions/distilled.csv"

# Checkpointing
CHECKPOINT_INTERVAL = 50