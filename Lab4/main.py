import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from model import RNN
from datasets import build_shopping
from data import load_data

def train(model, optimizer, criterion, train_loader, valid_loader, num_epochs):

    # initialize lists for storing loss and accuracy values
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in range(num_epochs):
        # training loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            labels, inputs = batch
            # labels, inputs = data['cat'].to(device), data['review'].to(device)

            inputs_size = inputs.size()  # 使用 .size() 方法获取各个维度的大小
            batch_size = inputs_size[0]
            max_sequence_length = inputs_size[1]
            embedding_size = inputs_size[2]

            print("Batch Size:", batch_size)
            print("Max Sequence Length:", max_sequence_length)
            print("Embedding Size:", embedding_size)

            # if len(inputs.shape) < 3:
            #     inputs = inputs[None]  # expand for batchsz

            # zero the parameter gradients
            optimizer.zero_grad()


            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # record loss and accuracy
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # calculate average train loss and accuracy
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # validation loop
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        for batch in valid_loader:
            labels, inputs = batch
            # if len(inputs.shape) < 3:
            #     inputs = inputs[None]  # expand for batchsz
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # record loss and accuracy
            valid_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            valid_correct += (predicted == labels).sum().item()
            valid_total += labels.size(0)

        # calculate average validation loss and accuracy
        valid_loss /= len(valid_loader.dataset)
        valid_acc = valid_correct / valid_total
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # print training status
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

    return train_losses, valid_losses, train_accs, valid_accs


def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(output, 1)
            total_correct += torch.sum(predictions == labels)

    test_loss = total_loss / len(test_loader.dataset)
    test_accuracy = total_correct.double() / len(test_loader.dataset)
    print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_accuracy))
    return test_loss, test_accuracy

# def train(model, data_loader, optimizer, criterion, epoch, device, dataset_name):
#     model.train()
#     train_loss = 0.0
#     all_loss = []
#     for idx, data in enumerate(data_loader):
#         label, sentence = data['cat'].to(device), data['review'].to(device)
#         print(sentence.shape)
#         if len(sentence.shape) < 3:
#             sentence = sentence[None]  # expand for batchsz
#         # print(sentence.shape)  # 1, words, 128
#         output = model(sentence)
#
#         optimizer.zero_grad()
#         if dataset_name == 'climate':
#             loss = abs(output - label)
#         else:
#             loss = criterion(output, label)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#         if idx % 376 == 0 and idx > 0:
#             all_loss.append(train_loss / 376)
#             print('[epoch %d, batch %d] loss: %.3f' % (epoch, idx, train_loss / 376))
#             train_loss = 0.0
#     return all_loss
#
# def validation(model, data_loader, device):
#     model.eval()
#     model.zero_grad()
#     correct_num = 0
#     val_num = 0
#     with torch.no_grad():
#         for data in tqdm(data_loader):
#             label, sentence = data['cat'].to(device), data['review'].to(device)
#             output = model(sentence)
#             pred = torch.argmax(output, dim=1)
#             correct_num += (pred == label).sum().item()
#             val_num += 1
#     print('Accuracy on validation set: %.2f%%\n' % (100.0 * correct_num / val_num))
#     return correct_num / val_num



if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_loader, val_loader, test_loader = load_data()
    # print(test_loader)
    num_epochs = 10

    # 定义模型参数
    input_size = 128
    hidden_size = 128
    output_size = 10

    model = RNN(device, input_size, hidden_size, output_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, criterion, train_loader, val_loader, num_epochs)

    # test(model, test_loader, criterion, device)








