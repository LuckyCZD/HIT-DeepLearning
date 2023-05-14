import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
from models.repvgg import create_RepVGG_A0
import torch.nn as nn
import pandas as pd
classes = ('Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed',
           'Common wheat', 'Fat Hen', 'Loose Silky-bent',
           'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet')
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std= [0.18507297, 0.18050247, 0.16784933])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_RepVGG_A0(deploy=True)
num_ftrs = model.linear.in_features
model.linear = nn.Linear(num_ftrs, 12)
model.load_state_dict(torch.load('repvgg_deploy.pth'))

model.eval()
model.to(DEVICE)

path = 'test/'
testList = os.listdir(path)
predList = []
for file in testList:
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    predList.append(classes[pred.data.item()])
    # print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
outputs = pd.DataFrame({'file': testList, 'species': predList})
outputs.to_csv(r"predicted.csv", index=False)  # index=False 代表不保存索引