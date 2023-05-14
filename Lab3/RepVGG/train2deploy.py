from models.repvgg import repvgg_model_convert
import torch
train_path='checkpoints/RepVgg/model_52_92.105.pth'
train_model=torch.load(train_path)
deploy_model = repvgg_model_convert(train_model, save_path='repvgg_deploy.pth')