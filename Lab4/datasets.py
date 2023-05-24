import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from torch.utils.data import DataLoader, Dataset
import torchtext.vocab as vocab
from torch.nn.utils.rnn import pad_sequence

shopping_cats = {'书籍': 0, '平板': 1, '手机': 2, '水果': 3, '洗发水': 4, '热水器': 5, '蒙牛': 6, '衣服': 7, '计算机': 8, '酒店': 9}  # 全部类别

class Shopping(Dataset):
    def __init__(self, my_list):
        self.cats = []
        self.reviews = []
        for i in range(len(my_list)):
            self.cats.append(my_list[i]['cat'])
            self.reviews.append(my_list[i]['review'])

    def __getitem__(self, idx):
        # return {"cat": self.cats[idx], "review": self.reviews[idx]}
        return {'cat': self.cats[idx], 'review': torch.tensor(np.array(self.reviews[idx]))}

    def __len__(self):
        return len(self.cats)


class Climate(Dataset):
    def __init__(self, my_data, my_label):
        sentence = []
        for i in range(my_data.shape[0]):
            word = []
            for j in range(0, 5):
                word.append(int(my_data.iloc[i, j]))
            sentence.append(word)
        self.data = np.array(sentence)
        self.label = my_label

    def __getitem__(self, idx):
        attr = self.data[idx]
        label = float(self.label.iloc[idx])
        return {'cat': label, 'review': attr}  # 和上面保持一致

    def __len__(self):
        return len(self.label)


def build_shopping():
    # 读取数据
    df = pd.read_csv("online_shopping_10_cats.csv", encoding='ANSI')
    val_list = []
    test_list = []
    train_list = []

    reviews = []
    cats = []
    for index, row in df.iterrows():
        if not isinstance(row['review'], str):
            continue
        cats.append(shopping_cats[row['cat']])
        reviews.append(row['review'])
    tokens = [jieba.lcut(i) for i in reviews]  # 分词

    model = Word2Vec(tokens, min_count=1, hs=1, window=3, vector_size=128)
    reviews_vector = [[model.wv[word] for word in sentence] for sentence in tokens]  # 转换成vector的reviews

    # 划分数据集
    for i in range(62773):
        if i % 5 == 4:
            val_list.append({'cat': cats[i], 'review': reviews_vector[i]})
        elif i % 5 == 0:
            test_list.append({'cat': cats[i], 'review': reviews_vector[i]})
        else:
            train_list.append({'cat': cats[i], 'review': reviews_vector[i]})

    # 对句子进行填充
    train_data = Shopping(train_list)
    val_data = Shopping(val_list)
    test_data = Shopping(test_list)

    collate_fn = lambda batch: (pad_sequence([torch.tensor(data['review']) for data in batch], batch_first=True),
                                torch.tensor([data['cat'] for data in batch]))


    batch_size = 64  # 设置批处理大小
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    print("finish")

    return train_loader, val_loader, test_loader
