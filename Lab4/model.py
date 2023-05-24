import torch
import torch.nn as nn

# class RNN(nn.Module):
#     def __init__(self, device, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.device = device
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
#         self.h2o = nn.Linear(self.hidden_size, output_size)
#         self.tanh = nn.Tanh()
#
#     def forward(self, x, hidden=None):  # x是一个句子，tensor
#         global output
#         if not hidden:
#             hidden = torch.zeros(1, self.hidden_size).to(self.device)
#         x = x[0]
#         for i in range(x.shape[0]):
#             token = x[i: i + 1]
#             combined = torch.cat((token, hidden), 1)
#             hidden = self.tanh(self.i2h(combined))
#             output = self.h2o(hidden)
#         return output


class RNN(nn.Module):
    def __init__(self, device, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.i2h = nn.Linear(self.input_size + self.hidden_size * 64, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden=None):
        global output
        batch_size = x.size(0)  # 获取批处理数据的大小
        if not hidden:
            hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
        for i in range(x.shape[1]):  # 修改为x的第二维度
            token = x[:, i: i + 1]  # 修改为使用批处理数据的切片
            combined = torch.cat((token, hidden), 1)
            hidden = self.tanh(self.i2h(combined))
            output = self.h2o(hidden)
        return output
