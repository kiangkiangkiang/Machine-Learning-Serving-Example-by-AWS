import json
import os
import re

from base_config import entity_type
from paddlenlp import Taskflow


# 必須包含的方法
## 1. 載入參數：load checkpoints with fixed-name function called 'model_fn'
def model_fn(model_dir: str):
    print("hahaha: in model_fn")
    print("model_dir: ", model_dir)
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
    result = []
    print("hahaha: in input_fun")
    print("request_body: ", request_body)
    data = json.loads(request_body)["inputs"]
    print("data: ", data)

    return data


## 3. 過模型
def predict_fn(input_object, model):
    print("hahaha in predict_fn")
    print("input_object: ", input_object)
    predict_result = model(input_object)
    print("predict_result: ", predict_result)
    return predict_result


## 4. 後處理
def output_fn(prediction, content_type):
    print("hahaha: in output_fn")
    return json.dumps(prediction)


"""
def inference(
    data_file: str,
    schema: List[str],
    device_id: int = 0,
    text_list: List[str] = None,
    precision: str = "fp32",
    batch_size: int = 1,
    model: str = "uie-base",
    task_path: str = None,
    postprocess_fun: Callable = lambda x: x,
    preprocess_fun: Callable = lambda x: x,
):
    if not os.path.exists(data_file) and not text_list:
        raise ValueError(f"Data not found in {data_file}. Please input the correct path of data.")

    if task_path:
        if not os.path.exists(task_path):
            raise ValueError(f"{task_path} is not a directory.")

        uie = Taskflow(
            "information_extraction",
            schema=schema,
            task_path=task_path,
            precision=precision,
            batch_size=batch_size,
            device_id=device_id,
        )
    else:
        uie = Taskflow(
            "information_extraction",
            schema=schema,
            model=model,
            precision=precision,
            batch_size=batch_size,
            device_id=device_id,
        )

    if not text_list:
        with open(data_file, "r", encoding="utf8") as f:
            text_list = [line.strip() for line in f]

    return postprocess_fun([uie(preprocess_fun(text)) for text in tqdm(text_list)])


if __name__ == "__main__":
    parser = PdArgumentParser((InferenceDataArguments, InferenceStrategyArguments, InferenceTaskflowArguments))
    data_args, strategy_args, taskflow_args = parser.parse_args_into_dataclasses()

    uie_processer = Processer(
        select_strategy=strategy_args.select_strategy,
        threshold=strategy_args.select_strategy_threshold,
        select_key=strategy_args.select_key,
        is_regularize_data=data_args.is_regularize_data,
    )

    logger.info("Start Inference...")

    inference_result = inference(
        data_file=data_args.data_file,
        device_id=taskflow_args.device_id,
        schema=entity_type,
        text_list=data_args.text_list,
        precision=taskflow_args.precision,
        batch_size=taskflow_args.batch_size,
        model=taskflow_args.model,
        task_path=taskflow_args.task_path,
        postprocess_fun=uie_processer.postprocess,
        preprocess_fun=uie_processer.preprocess,
    )

    logger.info("========== Inference Results ==========")
    for i, text_inference_result in enumerate(inference_result):
        logger.info(f"========== Content {i} Results ==========")
        logger.info(text_inference_result)
    logger.info("End Inference...")

    if data_args.save_dir:
        out_result = []
        if not os.path.exists(data_args.save_dir):
            logger.warning(f"{data_args.save_dir} is not found. Auto-create the dir.")
            os.makedirs(data_args.save_dir)

        with open(data_args.data_file, "r", encoding="utf8") as f:
            text_list = [line.strip() for line in f]

        with open(os.path.join(data_args.save_dir, data_args.save_name), "w", encoding="utf8") as f:
            for content, result in zip(text_list, inference_result):
                out_result.append(
                    {
                        "Content": content,
                        "InferenceResults": result,
                    }
                )
            jsonString = json.dumps(out_result, ensure_ascii=False)
            f.write(jsonString)
"""
