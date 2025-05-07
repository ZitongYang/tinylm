import torch
from typing import Dict
import numpy as np
from postrain import get_tokenizer

class MemmapDataset(torch.utils.data.Dataset):
    """
    A dataset that loads a memmap file.
    """
    def __init__(self, block_size: int, bin_file: str):
        self.block_size = block_size
        self.ids = np.memmap(bin_file, dtype=np.int32, mode='r')

    def __len__(self):
        return int(len(self.ids)/self.block_size)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i*self.block_size
        end_ind = (i+1)*self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        return dict(input_ids=torch.from_numpy(x_id).long(),
                    labels=torch.from_numpy(x_id).long())

def get_cpt_data():
    train = MemmapDataset(block_size=512,
                          bin_file='postrain.bin')
    return dict(train_dataset=train, eval_dataset=None)

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    block_size = 512
    data_module = get_cpt_data()
    for i in range(len(data_module['train_dataset'])):
        example = data_module['train_dataset'][i]
        print(tokenizer.decode(example['input_ids'][:6000]))
        import pdb; pdb.set_trace()