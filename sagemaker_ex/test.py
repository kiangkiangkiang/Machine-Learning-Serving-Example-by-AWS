# import json
# import os
# from functools import partial
# from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# from paddlenlp.datasets import load_dataset
# from paddlenlp.transformers import AutoTokenizer

"""
tokenizer = AutoTokenizer.from_pretrained("uie-base")
train_ds = load_dataset(read_data_by_chunk, data_path="./test.json", max_seq_len=512)

trans_fn = partial(
    convert_to_uie_format,
    tokenizer=tokenizer,
    max_seq_len=512,
)

train_ds = train_ds.map(trans_fn)
breakpoint()
"""

with open("s3://lab-luka-test/sagemaker-ex-local/label_studio_output.json", "r", encoding="utf-8") as f:
    s = f.read()
    breakpoint()
