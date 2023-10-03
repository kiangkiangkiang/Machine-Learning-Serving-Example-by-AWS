import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


@dataclass
class RecConfig:
    fc_features_sizes: List[int] = field(default_factory=lambda: [512, 768, 768, 512])
    fc_in_features: int = field(
        default=768,
        metadata={"help": "Number of fully connected layer"},
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    do_export: bool = field(default=False, metadata={"help": "Whether to export infernece model."})


class MLP(nn.Layer):
    def __init__(self, config, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.fc_layers = []
        act = F.ReLU()

        for i in range(len(config.fc_features_sizes) - 1):
            linear = paddle.nn.Linear(
                in_features=config.fc_features_sizes[i],
                out_features=config.fc_features_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(config.fc_features_sizes[i]))
                ),
            )
            self.add_sublayer(f"linear_{i}")
            self.fc_layers.append(linear)
            self.add_sublayer(f"activation_{i}")
            self.fc_layers.append(act)

    def forward(self, inputs: List[float]) -> List[float]:
        for fc in self.fc_layers:
            inputs = fc(inputs)
        return inputs


class TwoTower(nn.Layer):
    def __init__(self, config, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.forward = self.infer_forward

        # TODO user tower

        # TODO item tower

    def train(self):
        super().train()
        self.forward = self.train_forward

    def eval(self):
        super().eval()
        self.forward = self.infer_forward

    def train_forward(self):
        pass

    def infer_forward(self):
        pass


if __name__ == "__main__":
    breakpoint()
    my_config = RecConfig()
    my_mlp = MLP(my_config)
    people = 50
    variables = 512
    breakpoint()
    features = np.random.rand(people, variables)
    breakpoint()
    s = my_mlp(features)
    print(s)
    breakpoint()
