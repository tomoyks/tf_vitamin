from dataclasses import dataclass, asdict
from typing import List

import tensorflow as tf

from importlib import import_module



@dataclass
class ModelParameter:
    model_name: str = 'EfficientNetB7'
    optimizer: str = 'Adam'
    learning_rate: float = 0.1
    loss_fn: str = 'CategoricalCrossentropy'
    label_smoothing: float = 0.0001
    _metrics: str = 'categorical_accuracy'

    @property
    def metrics(self) -> List[str]:
        return self._metrics.split(",")

if __name__ == '__main__':
    model_param = ModelParameter()
