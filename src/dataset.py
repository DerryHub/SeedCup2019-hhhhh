import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class MyDataset(Dataset):
    def __init__(self, root='data', mode='train'):
        assert mode in ['train', 'evaluate', 'test', 'test_new'
                        ], mode + ' is not in ' + str(['train', 'evaluate'])
        df = pd.read_csv(os.path.join(root, mode + '.csv'))
        self.dic = df.to_dict(orient='list')
        self.keys = list(self.dic.keys())
        #for k in self.keys:
        #    self.dic[k] = self.dic[k][:100000]

    def __getitem__(self, index):
        d = {}
        for k in self.keys:
            d[k] = self.dic[k][index]
        return d

    def __len__(self):
        return len(self.dic[self.keys[0]])


if __name__ == "__main__":
    # dataset = MyDataset()
    loader = DataLoader(MyDataset(mode='train'), batch_size=10)
    for d in loader:
        print(d)
        break
