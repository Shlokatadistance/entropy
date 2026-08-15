from dataclasses import dataclass


@dataclass
class EntropyStats:
    Entropy: float
    EntropyNormalized: float
    TotalObjectCount: int
    TotalObjectTypes: int
    TopCommonObjects: list[tuple[str, int]]
