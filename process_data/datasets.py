
import os
from pathlib import Path

DATA_PATH = "/home/itmo/datasets/"
DATASETS = {p: Path(DATA_PATH + p) for p in os.listdir(DATA_PATH)}
