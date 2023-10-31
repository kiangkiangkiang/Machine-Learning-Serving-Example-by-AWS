# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback

import flask
import pandas as pd
from paddlenlp import Taskflow

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

try:
    os.system("export LD_LIBRARY_PATH=/opt/conda/lib")
    os.system("ls /usr/lib |grep lib")

except Exception as e:
    print("Customization: " + str(e))

entity_type = ["精神慰撫金額", "醫療費用", "薪資收入"]

# 必須包含的方法
## 1. 載入參數：load checkpoints with fixed-name function called 'model_fn'
def model_fn(model_dir: str):
    task_path = os.path.join(model_dir, "model_best")

    model = Taskflow(
        "information_extraction",
        schema=entity_type,
        task_path=task_path,
        precision="fp32",
    )

    print("model: ", model)
    return model


## 2. input 前處理
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"

    print("request_body: ", request_body)
    try:
        data = json.loads(request_body)["inputs"]
    except Exception as e:
        print("CustomizeMsg: ", str(e))
    print("data: ", data)

    return data


## 3. 過模型
def predict_fn(input_object, model):
    predict_result = ""
    try:
        predict_result = model(input_object)
    except Exception as e:
        print("CustomizeMsg: ", str(e))
    print("predict_result: ", predict_result)
    return predict_result


## 4. 後處理
def output_fn(prediction, content_type):
    print("hahaha: in output_fn")
    return json.dumps(prediction)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    health = 1
    status = 200 if health else 404
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def my_invoke_function():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    print("================ INVOCATIONS =================")
    print("<<<< flask.request.content_type", flask.request.content_type)
    if flask.request.content_type == "application/json":
        print("in my_invoke_function")
        data = flask.request.data.decode("utf-8")
        try:
            model = model_fn("/opt/program/")

            data = input_fn(data, content_type="application/json")
            result = predict_fn(data, model)
            result_json = output_fn(result, content_type="application/json")
        except Exception as e:
            result_json = {"error": str(e)}
            print("CustomizeMsg: ", str(e))

        return flask.Response(response=result_json, status=200, mimetype="application/json")
