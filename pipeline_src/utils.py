import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import cast, nn
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.utils.log import logger

loss_function = nn.BCELoss()

entity_type = ["精神慰撫金額", "醫療費用", "薪資收入"]

logger.set_level("INFO")

regularized_token = [r"\n", r" ", r"\u3000", r"\\n"]


def read_data_by_chunk(data_path: str, max_seq_len: int = 512) -> Iterator[Dict[str, str]]:
    """
    Summary: 讀「透過 utils/split_labelstudio.py 分割的 .txt檔」，此 txt 檔格式和 UIE官方提供的doccano.py轉換後的格式一樣。
        Model Input Format: [CLS] Prompt [SEP] Content [SEP].
        Result-Cross case: result cross interval of each subcontent.

    Args:
        data_path (str): 資料路徑（轉換後的training/eval/testing資料）。
        max_seq_len (int, optional): 模型input最大長度. Defaults to 512.

    Raises:
        ValueError: max_seq_len太小或prompt太長。
        DataError: 原始資料有問題（output of label studio），可能是entity太長或end的位置 < start的位置。

    Yields:
        Iterator[Dict[str, str]]: 每個batch所吃的原始文本（Before tokenization）。
    """

    if not os.path.exists(data_path):
        raise ValueError(f"Path not found {data_path}.")

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line["content"].strip()
            prompt = json_line["prompt"]

            # 3 means '[CLS] [SEP] [SEP]' in [CLS] Prompt [SEP] Content [SEP]
            if max_seq_len <= len(prompt) + 3:
                raise ValueError("The value of max_seq_len is too small. Please set a larger value.")
            result_list = json_line["result_list"]
            accumulate_token = 0

            # start pop all subcontent (segment by max_seq_len)
            while len(content) > 0:
                max_content_len = max_seq_len - len(prompt) - 3
                current_content_result = []

                # pop result in subcontent
                while len(result_list) > 0:
                    if (
                        result_list[0]["start"] > result_list[0]["end"]
                        or result_list[0]["end"] - result_list[0]["start"] > max_content_len
                    ):
                        raise ValueError(
                            f"Error in result list. Invalid start or end location (start: {result_list[0]['start']}, end: {result_list[0]['end']}). Please check the data in {data_path}."
                        )
                    if result_list[0]["start"] < max_content_len:
                        if result_list[0]["end"] > max_content_len:
                            # Result-Cross case: using dynamic adjust max_content_len to escape the problem.
                            logger.debug(f"Result-Cross. result: {result_list[0]}.")
                            max_content_len = result_list[0]["start"]
                            result_list[0]["start"] -= max_content_len
                            result_list[0]["end"] -= max_content_len
                            break
                        else:
                            current_content_result.append(result_list.pop(0))
                            if result_list:
                                result_list[0]["start"] -= accumulate_token
                                result_list[0]["end"] -= accumulate_token
                    else:
                        result_list[0]["start"] -= max_content_len
                        result_list[0]["end"] -= max_content_len
                        break

                for each_result in current_content_result:
                    adjust_data = content[:max_content_len][each_result["start"] : each_result["end"]]
                    true_data = each_result["text"]
                    if adjust_data != true_data:
                        raise ValueError(f"adjust error. adjust_data: {adjust_data}, true_data: {true_data}.")

                yield {
                    "content": content[:max_content_len],
                    "result_list": current_content_result,
                    "prompt": prompt,
                }

                content = content[max_content_len:]
                accumulate_token += max_content_len


def drift_offsets_mapping(offset_mapping: Tuple[Tuple[int, int]]) -> Tuple[List[List[int]], int]:
    """Scale the offset_mapping in tokenization output to align with the prompt learning format.

    Note: 因為 tokenization 後有些字會被 tokenize 在一起，所以 index 會和原本的有所差異，因此需做調整，將 tokenize 前後的 index 對齊。

    Args:
        offset_mapping (Tuple[Tuple[int, int]]): Tokenization outpu. Use argument 'return_offsets_mapping=True'.

    Returns:
        1. List[List[int, int]]: Scaled format, which is to adjust index after adding '[CLS] prompt [SEP]'.
        2. int: Drift term, which defines the scaling of drift after adjustment.
    """

    offset_mapping = [list(x) for x in offset_mapping]
    drift = 0
    for index in range(1, len(offset_mapping)):
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and drift == 0:
            drift = offset_mapping[index - 1][1] + 1  # [SEP] token
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += drift
        offset_mapping[index][1] += drift
    return offset_mapping, drift


def align_to_offset_mapping(origin_index: int, offset_mapping: List[List[int]]) -> int:
    """Align the original index (start/end index in result_list) to tokenized (offset_mapping) index.

    Args:
        origin_index (int): start/end index in result_list.
        offset_mapping (List[List[int, int]]): offset_mapping index after tokenization.

    Raises:
        PreprocessingError: Cannot find original index in offset_mapping.

    Returns:
        int: Aligned index.
    """
    final_index = 0
    for index, span in enumerate(offset_mapping):
        if span[0] <= origin_index < span[1]:
            return index
        if span[0] != 0 and span[1] != 0:
            final_index = index
    return final_index + 1


def convert_to_uie_format(
    data: Dict[str, str],
    tokenizer: Any,
    max_seq_len: int = 512,
    multilingual: Optional[bool] = False,
) -> Dict[str, Union[str, float]]:
    """此方法功能如下：
        1. Tokenization.
        2. 將 result_list 的 start/end index 對齊 tokenization 後的位置。

    Note:
        在 finetune.py 中，設定此方法為預設 Callback Function，可根據任務或模型換成自定義方法。

    ** Tokenize Bug **
        - Tokenizer 可能會因為中文的一些字 Unknown 導致 Bug (UIE Pretrained Model 並無此問題)。
        - 可參考：https://github.com/PaddlePaddle/PaddleNLP/issues?q=is%3Aissue+is%3Aopen+out+of+range+
        - 實測後可能有 Bug 的模型包含 xlnet, roformer.
        - 實測後正常的模型包含 uie, bert.

    Args:
        data (Dict[str, str], optional): 切片後的文本，通常來自於 () 的結果
            格式為 {"content": subcontent, "result_list": result_list_in_subcontent, "prompt": prompt}.
        tokenizer (Any, optional): paddlenlp.transformers.AutoTokenizer
        max_seq_len (int, optional): 切片文本的最大長度，通常與 () 一致，truncation 預設為 True. Defaults to 512.
        multilingual (Optional[bool], optional): Whether the model is a multilingual model. Defaults to False.

    Returns:
        Dict[str, Union[str, float]]: 模型真正的 input 格式。
    """
    if not data:
        return None

    try:
        encoded_inputs = tokenizer(
            text=[data["prompt"]],
            text_pair=[data["content"]],
            truncation=True,
            max_seq_len=max_seq_len,
            pad_to_max_seq_len=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_position_ids=True,
            return_dict=False,
            return_offsets_mapping=True,
        )[0]
    except Exception as e:
        logger.error(f"Tokenizer Error: {e}")
        logger.debug(f"Tokenizer Bug, content: {data['prompt']}")
        encoded_inputs = tokenizer(
            text=[data["prompt"]],
            text_pair=["無文本"],
            truncation=True,
            max_seq_len=max_seq_len,
            pad_to_max_seq_len=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_position_ids=True,
            return_dict=False,
            return_offsets_mapping=True,
        )[0]
        data["result_list"] = []

    start_ids, end_ids = map(lambda x: x * max_seq_len, ([0.0], [0.0]))

    # adjust offset_mapping
    adjusted_offset_mapping, drift = drift_offsets_mapping(offset_mapping=encoded_inputs["offset_mapping"])

    # align original index to tokenized (offset_mapping) index
    for item in data["result_list"]:
        aligned_start_index = align_to_offset_mapping(item["start"] + drift, adjusted_offset_mapping)
        aligned_end_index = align_to_offset_mapping(item["end"] - 1 + drift, adjusted_offset_mapping)
        start_ids[aligned_start_index] = 1.0
        end_ids[aligned_end_index] = 1.0

    return {
        "input_ids": encoded_inputs["input_ids"],
        "token_type_ids": encoded_inputs["token_type_ids"],
        "position_ids": encoded_inputs["position_ids"],
        "attention_mask": encoded_inputs["attention_mask"],
        "start_positions": start_ids,
        "end_positions": end_ids,
    }


def create_data_loader(dataset, mode="train", batch_size=16, trans_fn=None, shuffle=False):
    """
    Create dataloader.
    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        sampler = DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_sampler=sampler, return_list=True)
    return dataloader


def set_seed(seed: int) -> None:
    """設定種子

    Args:
        seed (int): 固定種子
    """

    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def shuffle_data(data: list) -> list:
    """shuffle data"""

    indexes = np.random.permutation(len(data))
    return [data[i] for i in indexes]


def convert_format(dataset: List[dict], entity_type: List[str], is_shuffle: bool = True) -> List[dict]:
    """轉換格式邏輯程式，將 label studio output 轉換成 UIE 模型所吃的格式。

    Args:
        dataset (List[dict]): label studio output的json檔。
        entity_type (List[str]): label studio 標注時所定義的所有 entity，例如「薪資收入」。
        is_shuffle (bool, optional): 是否隨機打亂資料. Defaults to True.

    Raises:
        ValueError: 任務格式錯誤（此邏輯程式只處理 NER 任務，有 Relation 標籤無法處理）。

    Returns:
        List[dict]: 模型所吃的訓練格式。
    """

    logger.info(f"Length of converting data: {len(dataset)}...")
    results = []
    for data in dataset:
        uie_format = {
            output_type: {"content": data["data"]["text"], "result_list": [], "prompt": output_type}
            for output_type in entity_type
        }
        for label_result in data["annotations"][0]["result"]:
            if label_result["type"] != "labels":
                raise ValueError(
                    "Now we only deal with NER tasks, "
                    "which means the type of label studio result is 'labels'. Please fix the input data type."
                )

            uie_format[label_result["value"]["labels"][0]]["result_list"].append(
                {
                    "text": label_result["value"]["text"],
                    "start": label_result["value"]["start"],
                    "end": label_result["value"]["end"],
                }
            )
        results.extend(uie_format.values())
    return shuffle_data(results) if is_shuffle else results


def read_json(json_file: str) -> None:
    """Read JSON files.

    Args:
        json_file (str): JSON files path and name.

    Raises:
        ValueError: File not found.

    """

    if os.path.exists(json_file):
        result = []
        with open(json_file, "r", encoding="utf-8") as infile:
            for f in infile:
                all_content = json.loads(f)
                for content in all_content:
                    result.append(content)
        return result
    else:
        raise ValueError(f"Cannot found the path {json_file}")


def regularize_content(
    single_json: str,
    regularize_text: Optional[List[str]] = ["\n", " ", "\u3000"],
    special_result_case: Optional[List[str]] = [r"\\n"],
) -> dict:
    """Regularize the keys of 'content' in the JSON file.
        The content may have several special tokens such as '\n', which does not want to be exist in the content.
        Therefore, this function will remove the special tokens and adjust the relatively index in the JSON file.

    Args:
        single_json (str): A JSON file which wants to be regularized. This file must be the same format with label studio output, and have the labels of NER tasks such as 'start' and 'end' index.
        regularize_text ((Optional[List[str]], optional): List of the special tokens. Each string in the list must be only one character (len(TOKEN) == 1).  Defaults to ["\n", " ", "\u3000"].
        special_result_case (Optional[List[str]], optional): Other special case which cannot be regularize in regularize_text. Defaults to [r"\n"].

    Returns:
        dict: Regularized file.
    """

    tmp = ""
    for i in regularize_text:
        tmp = tmp + i + "|"
    pattern = re.compile(tmp[:-1])

    if len(single_json["annotations"][0]["result"]) > 0:
        result_index = []
        # sorted result
        single_json["annotations"][0]["result"] = sorted(
            single_json["annotations"][0]["result"], key=lambda item: item["value"]["start"]
        )
        for i in single_json["annotations"][0]["result"]:
            result_index.append(i["value"]["start"])
            result_index.append(i["value"]["end"])

        logger.debug(f"result_index = {result_index}")

        # count the scale for result index
        special_token_counter = 0
        result_index_pointer = 0
        result_index_len = len(result_index)
        for i, char in enumerate(single_json["data"]["text"]):

            if i == result_index[result_index_pointer]:
                result_index[result_index_pointer] -= special_token_counter
                result_index_pointer += 1
                if result_index_pointer == result_index_len:
                    break
            if char in regularize_text:
                special_token_counter += 1

        # adjust result index
        result_index_pointer = 0
        for i in range(len(single_json["annotations"][0]["result"])):
            single_json["annotations"][0]["result"][i]["value"]["start"] = result_index[result_index_pointer]
            single_json["annotations"][0]["result"][i]["value"]["end"] = result_index[result_index_pointer + 1]
            single_json["annotations"][0]["result"][i]["value"]["text"] = re.sub(
                pattern, "", single_json["annotations"][0]["result"][i]["value"]["text"]
            )
            for u in special_result_case:
                single_json["annotations"][0]["result"][i]["value"]["text"] = re.sub(
                    u, "", single_json["annotations"][0]["result"][i]["value"]["text"]
                )

            result_index_pointer += 2

    single_json["data"]["text"] = re.sub(pattern, "", single_json["data"]["text"])
    return single_json


def regularize_json_file(
    json_file: str,
    out_variable: bool = False,
    output_path: str = "./",
    regularize_text: Optional[List[str]] = ["\n", " ", "\u3000"],
    special_result_case: Optional[List[str]] = [r"\\n"],
) -> None:
    """Regularize the JSON file list.

    Args:
        json_file (str): A JSON file path which contains several JSONs which want to be regularized.
        out_variable (bool): Defaults to False. If the output is a python varialble not write a file.
        output_path (str, optional): The path of regularized JSON. Defaults to "./".
        regularize_text (Optional[List[str]], optional): List of the special tokens. Each string in the list must be only one character (len(TOKEN) == 1).  Defaults to ["\n", " ", "\u3000"].
        special_result_case (Optional[List[str]], optional): Other special case which cannot be regularize in regularize_text. Defaults to [r"\n"].
    """

    if not os.path.exists(json_file):
        raise ValueError(
            f"Label studio file not found in {json_file}. Please input the correct path of label studio file."
        )

    def test_regularized_data(regularized_data: dict) -> bool:
        for i in range(len(regularized_data["annotations"][0]["result"])):
            start = regularized_data["annotations"][0]["result"][i]["value"]["start"]
            end = regularized_data["annotations"][0]["result"][i]["value"]["end"]
            adjusted_data = regularized_data["data"]["text"][start:end]
            true_data = regularized_data["annotations"][0]["result"][i]["value"]["text"]
            if adjusted_data != true_data:
                raise ValueError(
                    f"adjusted_data: {adjusted_data} is not equal to true_data: {true_data}. start:end = {start}:{end}"
                )

    for i in regularize_text:
        if len(i) != 1:
            raise ValueError("Default special token in regularize_text takes only 1 char!")

    logger.info(f"Start regularize ...")

    json_list = read_json(json_file)
    result_list = []
    for each_json in json_list:
        regularized_data = regularize_content(
            each_json, regularize_text=regularize_text, special_result_case=special_result_case
        )
        test_regularized_data(regularized_data)
        result_list.append(regularized_data)

    logger.info(f"Finish regularize data...")

    if out_variable:
        return result_list
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, "regularized_data.json"), "w", encoding="utf-8") as outfile:
            jsonString = json.dumps(result_list, ensure_ascii=False)
            outfile.write(jsonString)


def uie_loss_func(outputs, labels):
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = cast(start_ids, "float32")
    end_ids = cast(end_ids, "float32")
    loss_start = loss_function(start_prob, start_ids)
    loss_end = loss_function(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss


def compute_metrics(p):
    metric = SpanEvaluator()
    start_prob, end_prob = p.predictions
    start_ids, end_ids = p.label_ids
    metric.reset()
    num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
    metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    metric.reset()
    return {"precision": precision, "recall": recall, "f1": f1}


@dataclass
class ConvertArguments:
    labelstudio_file: str = field(
        default="./data/label_studio_data/label_studio_output.json",
        metadata={"help": "The export file from label studio. Only support the JSON format."},
    )

    save_dir: str = field(
        default="./data/model_input_data/",
        metadata={"help": "The path of converted data (train/dev/test) that you want to save."},
    )

    seed: int = field(
        default=1000,
        metadata={"help": "Random seed for shuffle."},
    )

    split_ratio: float = field(
        default_factory=lambda: [0.8, 0.1, 0.1],
        metadata={
            "help": "The ratio of samples in datasets. [0.7, 0.2, 0.1] means 70% samples used for training, 20% for evaluation and 10% for testing."
        },
    )

    is_shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the labeled dataset. Defaults to True."},
    )

    is_regularize_data: bool = field(
        default=True,
        metadata={"help": "Whether to regularize data (remove special tokens likes \\n). Defaults to True"},
    )


@dataclass
class TrainModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default="uie-base",
        metadata={
            "help": "Path to pretrained model, such as 'uie-base', 'uie-tiny', "
            "'uie-medium', 'uie-mini', 'uie-micro', 'uie-nano', 'uie-base-en', "
            "'uie-m-base', 'uie-m-large', or finetuned model path."
        },
    )

    max_seq_len: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class TrainDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    dataset_path: str = field(
        default="./data/model_input_data/",
        metadata={"help": "Local dataset directory including train.txt, dev.txt and test.txt (optional)."},
    )

    train_file: str = field(
        default="train.txt",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    dev_file: str = field(
        default="dev.txt",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    test_file: str = field(
        default="test.txt",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    export_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the exported inference model."},
    )


@dataclass
class EvaluationArguments(TrainModelArguments):

    dev_file: str = field(
        default="./data/model_input_data/test.txt",
        metadata={"help": "The path of file that you want to evaluate."},
    )

    batch_size: int = field(
        default=16,
        metadata={"help": "Evaluation batch size."},
    )

    device: str = field(
        default="gpu",
        metadata={"help": "The device for evaluate."},
    )

    is_eval_by_class: bool = field(
        default=False,
        metadata={
            "help": "Precision, recall and F1 score are calculated for each class separately if this option is enabled."
        },
    )


@dataclass
class InferenceDataArguments:
    data_file: str = field(
        default="./data/model_infer_data/example.txt",
        metadata={"help": "The path of data that you wanna inference."},
    )

    text_list: List[str] = field(
        default=None,
        metadata={"help": "The path of data that you wanna inference."},
    )

    save_dir: str = field(
        default=None,
        metadata={"help": "The path where you wanna to save results of inference. If None, model won't write data."},
    )

    save_name: str = field(
        default="inference_results.txt",
        metadata={"help": "Name of the inference results file."},
    )

    is_regularize_data: bool = field(
        default=False,
        metadata={"help": "Whether to regularize data (remove special tokens likes \\n). Defaults to False"},
    )


@dataclass
class InferenceTaskflowArguments:

    device_id: int = field(
        default=0,
        metadata={
            "help": "The device id for the gpu. The defalut value is 0. If you want to use CPU, set the device_id to -1."
        },
    )

    precision: str = field(
        default="fp32",
        metadata={
            "help": "fp16 or fp32. Default 'fp32', which is slower than 'fp16'. If 'fp16' is applied and gpu is used, make sure your CUDA>=11.2 and cuDNN>=8.1.1."
            "If there is warning when using fp16, pip install onnxruntime-gpu onnx onnxconverter-common."
        },
    )

    batch_size: int = field(
        default=1,
        metadata={"help": "Inference batch size."},
    )

    model: str = field(
        default="uie-base",
        metadata={
            "help": "Pretrained model, such as 'uie-base', 'uie-tiny', "
            "'uie-medium', 'uie-mini', 'uie-micro', 'uie-nano'. Only applied when task_path is None."
        },
    )

    task_path: str = field(
        default=None,
        metadata={"help": "The checkpoint you want to use on inference."},
    )


@dataclass
class InferenceStrategyArguments:

    select_strategy: str = field(
        default="all",
        metadata={
            "help": "'all' or 'max' or 'threshold'. Strategy of getting results. max: Only get the max prob. result. all: Get all results. "
            "threshold: Ｇet the results whose prob. is larger than threshold. "
        },
    )

    select_strategy_threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold for probability. Only applied when select_strategy='threshold'. "},
    )

    select_key: List[str] = field(
        default_factory=lambda: ["text", "start", "end", "probability"],
        metadata={
            "help": "UIE will output ['text', 'start', 'end', 'probability']. --select_key is to select which key in the list you want to return."
        },
    )
