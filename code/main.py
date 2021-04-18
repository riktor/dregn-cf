import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from logger import getLogger, log_codes, log_config
from train import train

# ==============================
np.random.seed(2021)
random.seed(2021)
# ==============================


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    os.chdir(hydra.utils.get_original_cwd())

    timestamp = time.time()
    # logging config of the experiment
    log_config(cfg, timestamp)

    # logging freezed codes of the experiment
    log_codes(cfg, timestamp)

    logger = getLogger(cfg, timestamp)

    data_dir = Path(cfg.dataset_path)
    train(data_dir, cfg, logger)


if __name__ == "__main__":
    main()
