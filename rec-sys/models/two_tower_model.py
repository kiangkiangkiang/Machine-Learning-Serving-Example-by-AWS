# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
import math

import net
import numpy as np
import paddle
import paddle.nn.functional as F


# TODO 修改
class DygraphModel:
    # define model
    def create_model(self, config):
        sparse_feature_number = config.get("hyper_parameters.sparse_feature_number")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
        fc_sizes = config.get("hyper_parameters.fc_sizes")

        Recall_model = net.DNNLayer(sparse_feature_number, sparse_feature_dim, fc_sizes)
        return Recall_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data):
        user_sparse_inputs = [paddle.to_tensor(batch_data[i].numpy().astype("int64").reshape(-1, 1)) for i in range(4)]

        mov_sparse_inputs = [
            paddle.to_tensor(batch_data[4].numpy().astype("int64").reshape(-1, 1)),
            paddle.to_tensor(batch_data[5].numpy().astype("int64").reshape(-1, 4)),
            paddle.to_tensor(batch_data[6].numpy().astype("int64").reshape(-1, 3)),
        ]

        label_input = paddle.to_tensor(batch_data[7].numpy().astype("int64").reshape(-1, 1))

        return user_sparse_inputs, mov_sparse_inputs, label_input

    # define loss function by predicts and label
    def create_loss(self, predict, label_input):
        cost = F.square_error_cost(predict, paddle.cast(x=label_input, dtype="float32"))
        avg_cost = paddle.mean(cost)
        return avg_cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric

    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):

        batch_size = config.get("runner.train_batch_size", 128)
        user_sparse_inputs, mov_sparse_inputs, label_input = self.create_feeds(batch_data)

        predict = dy_model.forward(batch_size, user_sparse_inputs, mov_sparse_inputs, label_input)
        loss = self.create_loss(predict, label_input)
        # update metrics
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        batch_runner_result = {}
        batch_size = config.get("runner.infer_batch_size", 128)
        user_sparse_inputs, mov_sparse_inputs, label_input = self.create_feeds(batch_data)

        predict = dy_model.forward(batch_size, user_sparse_inputs, mov_sparse_inputs, label_input)
        # update metrics
        uid = user_sparse_inputs[0]
        movieid = mov_sparse_inputs[0]
        label = label_input
        predict = predict

        batch_runner_result["userid"] = uid.numpy().tolist()
        batch_runner_result["movieid"] = movieid.numpy().tolist()
        batch_runner_result["label"] = label.numpy().tolist()
        batch_runner_result["predict"] = predict.numpy().tolist()

        print_dict = {"predict": predict}
        return metrics_list, print_dict, batch_runner_result


# TODO 修改
class StaticModel:
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.sparse_feature_number = self.config.get("hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get("hyper_parameters.sparse_feature_dim")
        self.hidden_layers = self.config.get("hyper_parameters.fc_sizes")
        self.learning_rate = self.config.get("hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):
        userid = paddle.static.data(name="userid", shape=[-1, 1], dtype="int64")
        gender = paddle.static.data(name="gender", shape=[-1, 1], dtype="int64")
        age = paddle.static.data(name="age", shape=[-1, 1], dtype="int64")
        occupation = paddle.static.data(name="occupation", shape=[-1, 1], dtype="int64")
        user_sparse_inputs = [userid, gender, age, occupation]

        movieid = paddle.static.data(name="movieid", shape=[-1, 1], dtype="int64")
        title = paddle.static.data(name="title", shape=[-1, 4], dtype="int64")
        genres = paddle.static.data(name="genres", shape=[-1, 3], dtype="int64")
        mov_sparse_inputs = [movieid, title, genres]

        label_input = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

        feeds_list = user_sparse_inputs + mov_sparse_inputs + [label_input]
        return feeds_list

    def net(self, input, is_infer=False):
        self.user_sparse_inputs = input[:4]
        self.mov_sparse_inputs = input[4:7]
        self.label_input = input[-1]
        if is_infer:
            self.batch_size = self.config.get("runner.infer_batch_size")
        else:
            self.batch_size = self.config.get("runner.train_batch_size")
        recall_model = DNNLayer(self.sparse_feature_number, self.sparse_feature_dim, self.hidden_layers)
        predict = recall_model.forward(
            self.batch_size, self.user_sparse_inputs, self.mov_sparse_inputs, self.label_input
        )

        self.inference_target_var = predict
        if is_infer:
            uid = self.user_sparse_inputs[0]
            movieid = self.mov_sparse_inputs[0]
            label = self.label_input
            predict = predict
            fetch_dict = {"userid": uid, "movieid": movieid, "label": label, "predict": predict}
            return fetch_dict
        cost = F.square_error_cost(predict, paddle.cast(x=self.label_input, dtype="float32"))
        avg_cost = paddle.mean(cost)
        self._cost = avg_cost
        fetch_dict = {"Loss": avg_cost}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(learning_rate=self.learning_rate, lazy_mode=True)
        if strategy != None:
            import paddle.distributed.fleet as fleet

            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
