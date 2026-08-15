import gc
import math
from collections import Counter, defaultdict
from sys import getsizeof

from utils.models import EntropyStats


class ShannonEntropy:
    def __init__(self, use_bytes=False) -> None:
        self.use_bytes = use_bytes
        self.probabilites = []
        self.net_counts = Counter()
        self.stats = None

    def gather_objects(self) -> "ShannonEntropy":
        gc.collect()
        objs = gc.get_objects()
        weights = defaultdict(int)
        self.net_counts = Counter(type(obj).__name__ for obj in objs)
        if self.use_bytes:
            # Always ensure the calculation is on obj and on type(obj).__name__, which will always be a string
            for obj in objs:
                weights[type(obj).__name__] += getsizeof(obj)
            net_weight = sum(weights.values())
            self.probabilites = [k / net_weight for k in weights.values()]
        else:
            total = sum(self.net_counts.values())
            self.probabilites = [self.net_counts[k] / total for k in self.net_counts.keys()]
        return self

    def calculate_shannon_entropy(self) -> "ShannonEntropy":
        H = -sum(p * math.log2(p) for p in self.probabilites if p > 0)
        H_max = math.log2(len(self.probabilites)) if len(self.probabilites) else 0

        self.stats = EntropyStats(
            Entropy=H,
            EntropyNormalized=H / H_max if H_max else 0,
            TotalObjectCount=sum(self.net_counts.values()),
            TotalObjectTypes=len(self.net_counts.keys()),
            TopCommonObjects=self.net_counts.most_common(10),
        )
        return self
