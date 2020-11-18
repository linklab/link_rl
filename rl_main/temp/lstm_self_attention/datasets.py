import math

import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, window, target_cols):
        self.data = torch.Tensor(data)
        self.window = window
        self.target_cols = target_cols
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        x = self.data[index: index + self.window]
        y = self.data[index + self.window, self.target_cols]
        return x, y

    def __len__(self):
        return len(self.data) - self.window

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getsize__(self):
        return (self.__len__())


batch_size = 32
input_feature_size = 4
seq_length = 18
target_cols = 1
pin_memory = True
num_workers = 1

# data shape: (40000,)
data = torch.tensor([math.sin(x) for x in range(0, 10000 * input_feature_size)])

# data shape: (10000, 4)
data = data.view(10000, input_feature_size)

dataset = MyDataset(data, seq_length, target_cols)

total_data_size = len(dataset)
train_data_size = int(len(dataset) * 0.7)
valid_data_size = int(len(dataset) * 0.2)
test_data_size = total_data_size - train_data_size - valid_data_size

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_data_size, valid_data_size, test_data_size]
)

train_iter = DataLoader(
    train_dataset, batch_size=batch_size,
    drop_last=True,
    # num_workers=num_workers,
    pin_memory=pin_memory
)

valid_iter = DataLoader(
    valid_dataset, batch_size=batch_size,
    drop_last=True,
    # num_workers=num_workers,
    pin_memory=pin_memory
)

test_iter = DataLoader(
    test_dataset, batch_size=batch_size,
    drop_last=True,
    # num_workers=num_workers,
    pin_memory=pin_memory
)

# for idx_batch, batch in enumerate(train_iter):
#     X, Y = batch
#     print(idx_batch, X.shape, Y.shape)
#     print(X, Y)
#     print("\n\n\n")
