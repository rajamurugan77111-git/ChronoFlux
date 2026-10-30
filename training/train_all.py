"""Train every ChronoFlux model.

The current-state detectors (binary detection + attack-category stage) are
selected by the model arena, which benchmarks several algorithms and deploys
the best one for each task. The legacy single-model scripts
(train_detection.py / train_stage.py) are kept for reference.
"""
from training.model_arena import main as train_arena
from training.train_anomaly import main as train_anomaly
from training.train_world_model import main as train_world


if __name__ == "__main__":
    train_arena([])
    train_anomaly()
    train_world()
