import pandas as pd
import numpy as np
import jieba
import torch
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def load_data():
    # 读取数据集
    data = pd.read_csv("online_shopping_10_cats.csv", encoding='ANSI')
    reviews = data["review"].tolist()
    categories = data["cat"].tolist()

    # 获取数据集长度
    num_samples = len(reviews)

    # 构建索引列表
    indices = np.arange(num_samples)

    # 划分数据集
    train_indices = indices[(indices % 5) != 0]  # 训练集索引
    val_indices = indices[(indices % 5) == 4]  # 验证集索引
    test_indices = indices[(indices % 5) == 0]  # 测试集索引

    train_reviews = [reviews[i] for i in train_indices]
    val_reviews = [reviews[i] for i in val_indices]
    test_reviews = [reviews[i] for i in test_indices]

    train_categories = [categories[i] for i in train_indices]
    val_categories = [categories[i] for i in val_indices]
    test_categories = [categories[i] for i in test_indices]

    tokens = [jieba.lcut(i) for i in train_reviews]  # 分词

    # 构建词向量模型

    word2vec_model = Word2Vec(tokens, min_count=1, hs=1, window=3, vector_size=128)

    # 将评论转换为词向量序列
    train_sequences = [[word2vec_model.wv[word] for word in review] for review in tokens]

    # 创建数据加载器
    train_dataset = MyDataset(train_sequences, train_categories)
    val_dataset = MyDataset(val_reviews, val_categories)
    test_dataset = MyDataset(test_reviews, test_categories)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


class MyDataset(Dataset):
    def __init__(self, reviews, categories):
        self.reviews = reviews
        self.categories = categories

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = self.reviews[index]
        category = self.categories[index]

        # 将评论转换为张量
        review_tensor = torch.Tensor(review)

        # 返回评论张量和类别
        return review_tensor, category