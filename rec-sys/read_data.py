import numpy as np
import paddle
from paddle.io import DataLoader, IterableDataset
from paddlenlp.transformers import UIE

uie = UIE.from_pretrained("uie-nano")


with open("data/ml-1m/movies.dat", "r", encoding="utf8") as f:
    print(123)
    breakpoint()

"""
# get data
def get_data(batch_size=64, device="cpu", num_workers=0, file_dir, ):
    from importlib import import_module

    place = paddle.set_device(device)
    reader_class = import_module(reader_path)
    dataset = reader_class.RecDataset(file_list, config=config)
    loader = DataLoader(dataset, batch_size=batch_size, places=place, drop_last=True, num_workers=num_workers)
    return loader
"""


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list

    def __iter__(self):
        full_lines = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for line in rf:
                    output_list = []
                    features = line.strip().split(",")
                    user_input = [int(features[0])]
                    item_input = [int(features[1])]
                    label = [int(features[2])]
                    output_list.append(np.array(user_input).astype("int64"))
                    output_list.append(np.array(item_input).astype("int64"))
                    output_list.append(np.array(label).astype("int64"))
                    yield output_list
