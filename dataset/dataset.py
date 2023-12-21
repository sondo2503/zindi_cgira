from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import cv2

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class DefaultDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=5,
                 split_prefix='split.stratified',
                 **_):
        self.split = split
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.split_prefix = split_prefix
        self.images_dir = os.path.join(dataset_dir, 'images')
        # self.external_images_dir = os.path.join(dataset_dir, 'rgby', 'external')

        self.df_labels = self.load_labels()
        self.examples = self.load_examples()
        self.size = len(self.examples)

    def load_labels(self):
        labels_path = '{}.{}.csv'.format(self.split_prefix, self.idx_fold)
        labels_path = os.path.join(self.dataset_dir, labels_path)
        df_labels = pd.read_csv(labels_path)
        df_labels = df_labels[df_labels['Split'] == self.split]
        df_labels = df_labels.reset_index()

        def to_filepath(v):
            return os.path.join(self.images_dir, v)

        df_labels['filepath'] = df_labels['Id'].transform(to_filepath)
        return df_labels

    def load_examples(self):
        return [(row['Id'], row['filepath'], int(row['Target']))
                for _, row in self.df_labels.iterrows()]

    def __getitem__(self, index):
        example = self.examples[index]

        filename = example[1]
        image = cv2.imread(filename)

        label = [0 for _ in range(5)]
        l = example[2]
        label[l] = 1
        label = np.array(label)

        if self.transform is not None:
            image = self.transform(image)

        return {'image': image,
                'label': label,
                'key': example[0]}

    def __len__(self):
        return self.size


def test():
    dataset = DefaultDataset('../data', 'train', None)
    # print(len(dataset))
    example = dataset[4]
    example = dataset[4]

    dataset = DefaultDataset('../data', 'val', None)


def get_dataset(config, split, transform=None, last_epoch=-1):
    f = globals().get(config.name)

    return f(config.dir,
             split=split,
             transform=transform,
             **config.params)


def get_dataloader(config, split, transform=None, **_):
    dataset = get_dataset(config.data, split, transform)

    is_train = 'train' == split
    batch_size = config.train.batch_size if is_train else config.eval.batch_size

    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            drop_last=is_train,
                            num_workers=config.transform.num_preprocessor,
                            pin_memory=False)
    return dataloader


if __name__ == '__main__':
    test()