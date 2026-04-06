from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
