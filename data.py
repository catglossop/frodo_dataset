from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class TimestampedData:
    data: Any
    timestamps: np.ndarray


TimestampedDataDict = Dict[str, TimestampedData]