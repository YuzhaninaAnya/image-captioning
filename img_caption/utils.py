import torch
import random
import numpy as np


def collate_fn(batch):
    pad_idx = 0
    max_len = max([len(elem["caption"]) for elem in batch])
    pad_captions = torch.stack([
        torch.cat((elem["caption"], torch.LongTensor([pad_idx] * (max_len - len(elem["caption"])))))
        for elem in batch
    ])
    enc_img = torch.stack([elem["image"] for elem in batch])

    return enc_img, pad_captions


def init_random_seed(value=42):
    """
    Set seed for reproducibility
    :param value: seed
    """
    random.seed(value)
    np.random.seed(value)
    torch.random.manual_seed(value)
    torch.cuda.random.manual_seed(value)
    torch.cuda.random.manual_seed_all(value)


def split_data(data, train_split=0.8):
    """
    Split dataset into random train and test subsets
    :param data: dataset to be split
    :param train_split: represent the proportion of the dataset to include in the train split
    :return: train-test split of inputs
    """
    train_size = int(train_split * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data,
                                                                [train_size, test_size])

    return train_dataset, test_dataset


def softmax(x, temperature=5):
    """
    Applies a softmax function
    :param x: input
    :param temperature:
    """
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)
