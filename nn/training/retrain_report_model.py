"""Reproduce the OPNO champion model used in the report (W64_D1_OH64_OC64_FH64_P8,
originally trained under nn/training/benchmark/opno_search/full/) from its own
recorded model_meta.json, so the exact training config is not at the mercy of
whatever nn/training/config.py's TrainConfig defaults happen to be today.

Only two fields are overridden from the recorded config:
  - checkpoint_dir: a fresh directory, so the original benchmark checkpoint
    (used for the report's numbers/figures) is never touched.
  - plot_every: turned on (recorded run used 0, i.e. no snapshots), so this
    run produces the during-training DGSEM-vs-MUSCL snapshot images via
    training.viz_snapshot, using the restyled vizstyle/plotstyle.

Run from the repo root:
    .venv_spectral/bin/python nn/training/retrain_report_model.py
"""

import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import TrainConfig
from training.train import train, validate_config

SOURCE_META = ("nn/training/benchmark/opno_search/full/"
              "W64_D1_OH64_OC64_FH64_P8/model_meta.json")
CHECKPOINT_DIR = "nn/training/checkpoints_W64_D1_OH64_OC64_FH64_P8"
PLOT_EVERY = 25   # -> ~10 during-training snapshots over 250 epochs


def build_config():
    with open(SOURCE_META) as f:
        meta = json.load(f)
    cfg = TrainConfig(**meta["config"])
    cfg = replace(cfg, checkpoint_dir=CHECKPOINT_DIR, plot_every=PLOT_EVERY)
    validate_config(cfg)
    return cfg


if __name__ == "__main__":
    cfg = build_config()
    print(f"config (from {SOURCE_META}): {cfg}")
    train(cfg)
