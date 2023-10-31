import json
import os

try:
    os.system("export LD_LIBRARY_PATH=/opt/conda/lib")
    os.system("ls /usr/lib |grep lib")

except Exception as e:
    print("Customization: " + str(e))

import re

from base_config import entity_type
from paddlenlp import Taskflow


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
