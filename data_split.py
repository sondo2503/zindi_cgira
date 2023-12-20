import os
import argparse
import math
from itertools import combinations
import random

import tqdm
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def make_examples_dict(df):
    damage = {
        'DR': 0,
        'G': 1,
        'ND': 2,
        'WD': 3,
        'other': 4
    }
    examples_dict = {}
    for _, row in tqdm.tqdm(df.iterrows()):
        id_ = row['filename']
        labels = damage[row['damage']]
        examples_dict[id_] = labels
    return examples_dict


def split_stratified(all_examples_dict):
    examples = []
    y_list = []
    for key, labels in all_examples_dict.items():
        labels = [labels]
        np_labels = np.zeros((28,), dtype=int)
        np_labels[np.array(labels)] = 1
        examples.append((key, labels))
        y_list.append(np_labels)

    X = np.arange(len(y_list))
    y = np.array(y_list)

    # test_val
    mskf = MultilabelStratifiedKFold(n_splits=5, random_state=1234, shuffle=True)
    folds = []
    for train_index, test_index in mskf.split(X, y):
        folds.append(test_index)

    for a, b in combinations(folds, 2):
        assert len(set(a) & set(b)) == 0
    return examples, folds


def save(examples, folds, num_fold, data_dir):
    for fold_idx in range(num_fold):
        records = []
        for i, indices in enumerate(folds):
            if i == fold_idx:
                for j in indices:
                    records.append((examples[j][0], examples[j][1], 'val'))
            else:
                for j in indices:
                    records.append((examples[j][0], examples[j][1][0], 'train'))
        df = pd.DataFrame.from_records(records, columns=['Id', 'Target', 'Split'])
        output_filename = os.path.join(data_dir, 'split.stratified.{}.csv'.format(fold_idx))

        df.to_csv(output_filename, index=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', dest='data_dir',
                        help='the directory of the data',
                        default='data', type=str)
    # parser.add_argument('--use_external', dest='use_external',
    #                     help='1: with external, 0: without external',
    #                     default=1, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    num_fold = 5
    data_dir = args.data_dir

    df = pd.read_csv(os.path.join(data_dir, 'Train.csv'))
    examples_dict = make_examples_dict(df)

    examples, folds = split_stratified(examples_dict)
    save(examples, folds, num_fold, data_dir)
    # for i in folds:
    #     print(i)


if __name__ == '__main__':
    main()
