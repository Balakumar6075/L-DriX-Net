# ============================================================
# L-DriX-Net Temporal Configuration
# ============================================================

TEMPORAL_LENGTH = 16

# Frames processed per second
TEMPORAL_FPS = 5

# Temporal window duration
TEMPORAL_SECONDS = TEMPORAL_LENGTH / TEMPORAL_FPS

# Driver state classes
STATE_CLASSES = [
    "CONCENTRATED",
    "DISTRACTED",
    "DROWSY",
]

NUM_STATES = len(STATE_CLASSES)

# State training
STATE_BATCH_SIZE = 4
STATE_EPOCHS = 30

STATE_LEARNING_RATE = 1e-4
STATE_WEIGHT_DECAY = 1e-5

# Classification loss weight
STATE_LOSS_WEIGHT = 1.0

# Sliding-window stride
WINDOW_STRIDE = 1