import math

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from PaddleRec.models.recall.ncf.net import NCF_MLP_Layer


class MLP(nn.Layer):
    def __init__(self, num_objects, layers):
        self.num_objects = num_objects
        self.layers = layers
        super(MLP, self).__init__()
        self.MLP_Embedding = nn.Embedding(
            self.num_objects,
            int(self.layers[0]),
            sparse=True,
            weight_attr=nn.initializer.Normal(mean=0.0, std=0.01),
        )

        num_layer = len(self.layers)
        self.MLP_fc = []
        for i in range(1, num_layer):
            Linear = nn.Linear(
                in_features=self.layers[i - 1],
                out_features=self.layers[i],
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.TruncatedNormal(mean=0.0, std=1.0 / math.sqrt(self.layers[i - 1]))
                ),
                name="layer_" + str(i),
            )
            self.add_sublayer("layer_%d" % i, Linear)
            self.MLP_fc.append(Linear)
            act = paddle.nn.ReLU()
            self.add_sublayer("act_%d" % i, act)
            self.MLP_fc.append(act)

    def forward(self, input_data):
        input = input_data

        mlp_embedding = self.MLP_Embedding(input)
        mlp_embedding_latent = paddle.flatten(x=mlp_embedding, start_axis=1, stop_axis=2)

        for n_layer in self.MLP_fc:
            mlp_embedding_latent = n_layer(mlp_embedding_latent)

        return mlp_embedding_latent


class TwoTower(nn.Layer):
    def __init__(self, num_users, num_items, layers):
        super(TwoTower, self).__init__()
        self.user_tower = MLP(num_users, layers)
        self.item_tower = MLP(num_items, layers)

        self.prediction = paddle.nn.Linear(
            in_features=layers[3],
            out_features=1,
            weight_attr=nn.initializer.KaimingUniform(fan_in=layers[3] * 2),
            name="prediction",
        )

        # self.prediction = F.cosine_similarity()

        self.sigmoid = paddle.nn.Sigmoid()

    def train_forward(self, input_data):
        user_representation = self.user_tower(input_data[0])
        item_representation = self.item_tower(input_data[1])

        # mlp_vector = paddle.concat(x=[mlp_user_latent, mlp_item_latent], axis=-1)

        # prediction = self.prediction(mlp_vector)
        # prediction = self.prediction(user_representation, item_representation)
        prediction = F.cosine_similarity(user_representation, item_representation)
        # prediction = self.sigmoid(prediction)

        return prediction


people = 10000
variables = 512
features = np.random.rand(people, variables)
mymodel = TwoTower(num_users=50, num_items=555, layers=[512, 768, 768, 512])
breakpoint()


x = paddle.to_tensor([0, 2, 1, 4, 2], dtype="float32")
y = paddle.to_tensor([1, 2, 3, 4, 5], dtype="float32")
ss = F.cosine_similarity(x, y, axis=0)
breakpoint()
