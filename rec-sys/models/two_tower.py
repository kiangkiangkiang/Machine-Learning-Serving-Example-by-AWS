import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

"""
from paddlenlp.transformers import UIE

tmp = UIE.from_pretrained("uie-mini")

breakpoint()
"""


@dataclass
class MLPConfig:
    fc_features_sizes: List[int] = field(default_factory=lambda: [512, 768, 768, 512])
    fc_in_features: int = field(
        default=768,
        metadata={"help": "Number of fully connected layer"},
    )
    dropout_rate: float = field(
        default=0.2,
        metadata={"help": "Dropout rate of fully connected layer"},
    )


class MLP(nn.Layer):
    def __init__(self, config, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.fc_layers = []
        act = nn.ReLU()
        dropout_layer = nn.Dropout(p=config.dropout_rate)

        for i in range(len(config.fc_features_sizes) - 1):
            linear = nn.Linear(
                in_features=config.fc_features_sizes[i],
                out_features=config.fc_features_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(config.fc_features_sizes[i]))
                ),
            )
            layer_norm = nn.LayerNorm(config.fc_features_sizes[i + 1])
            self.add_sublayer(f"linear_{i}", linear)
            self.fc_layers.append(linear)
            self.add_sublayer(f"activation_{i}", act)
            self.fc_layers.append(act)
            self.add_sublayer(f"layer_norm_{i}", layer_norm)
            self.fc_layers.append(layer_norm)
            self.add_sublayer(f"dropout_{i}", dropout_layer)
            self.fc_layers.append(dropout_layer)

    def forward(self, inputs: List[float]) -> List[float]:
        for fc in self.fc_layers:
            inputs = fc(inputs)
        return inputs


class TwoTower(nn.Layer):
    def __init__(self, config, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.forward = self.infer_forward
        self.prediction = F.cosine_similarity

        # TODO fix config: features extraction
        self.user_tower = self.create_model()
        self.item_tower = self.create_model()

    def create_model(self):
        model_config = MLPConfig()
        mlp = MLP(model_config)
        return mlp

    def train_forward(self, user_inputs: List[float], item_inputs: List[float]):
        user_embed = self.user_tower(user_inputs)
        item_embed = self.item_tower(item_inputs)

        p = self.prediction(user_embed, item_embed, axis=1).reshape([-1, 1])
        return p

    def infer_forward(self, user_inputs: List[float]):
        user_embed = self.user_tower(user_inputs)
        return user_embed


if __name__ == "__main__":
    my_config = MLPConfig()
    my_mlp = MLP(my_config)
    people = 64
    variables = 512
    item = 64
    item_var = 512
    features = np.random.rand(people, variables)
    item_features = np.random.rand(item, item_var)
    features = paddle.to_tensor(features, dtype="float32")
    item_features = paddle.to_tensor(item_features, dtype="float32")
    # s = my_mlp(features)
    t = TwoTower(config=-1)
    t.train()
    s = t(features, item_features)
    print(s)
    breakpoint()
