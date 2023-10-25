import json


## 1. 載入模型
def model_fn(model_dir: str):
    """Sagemaker 預設會將上傳好的 s3 model.tar.gz 解壓縮，解壓縮路徑為 model_dir。
        此方法讓使用者自定義如何「載入模型」。
        例如：
            1.
                with open(os.path.join(model_dir, "model.pth"), "rb") as f:
                    model.load_state_dict(torch.load(f))
            2.
                paddle.load(os.path.join(model_dir, "emb.pdparams"))

    Args:
        model_dir (str): 解壓縮 S3 模型的路徑。

    Returns:
        _type_: 載入的可用於 Inference 的模型
    """

    print("CustomizeMsg: In model_fn.")
    print("model_dir: ", model_dir)
    return "My Model"


## 2. input 前處理
def input_fn(request_body, request_content_type):
    """前處理程式

    Args:
        request_body (_type_): 使用 sagemaker.pytorch.model.PyTorchPredictor predict 的 data。就是你輸入的原始資料。
            例如：
                dummy_data = {"test_input_data": [1,1,2,3]}
                predictor.predict(data=dummy_data)
                # 則 request_body 大致等同於 dummy_data (原始資料進來後可能會有編碼問題)

    Returns:
        _type_: 丟給模型的資料，也就是 predict_fn 內的 input_object。
    """

    print("CustomizeMsg: In input_fn.")
    print("request_body: ", request_body)

    return "data after input_fun"


## 3. 過模型
def predict_fn(input_object, model):
    """過模型 Inference 的程式

    Args:
        input_object (_type_): 前處理完的資料
        model (_type_): 主要負責 Inference 的模型（來自 model_fn 的結果）

    Returns:
        _type_: 模型預測結果
    """

    print("CustomizeMsg: In predict_fn.")
    print("input_object: ", input_object)

    return "predict result"


## 4. 後處理
def output_fn(prediction, content_type):
    """後處理程式

    Args:
        prediction (_type_): predict_fn 的回傳結果。

    Returns:
        _type_: 回傳值一定要是一個 byte array 並且和 content_type 指定的格式對齊。
            例如：content_type: application/json，則使用 json.dumps 包裝結果。

    """

    print("CustomizeMsg: In prediction.")
    print("prediction result: ", prediction)

    return json.dumps({"prediction result after output_fn": prediction})
