import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter

import os
import argparse
import numpy as np
import sys
import itertools
from tqdm import tqdm

best_acc = 0.0

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training Extration features')
parser.add_argument("--fold", required=True, type=str, help="folds")
parser.add_argument("--train", required=True, type=str, help="path train")
parser.add_argument("--val", required=False, type=str, help="path val")
parser.add_argument("--test", required=True, type=str, help="path test")
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--epoch', default=200, type=int, help='you need in the epoch')

args = parser.parse_args()

writer = SummaryWriter()

seed = "Resnet50_finetune_fold"+str(args.fold)

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train = os.path.abspath(args.train)
val = os.path.abspath(args.val)
test = os.path.abspath(args.test)


trainset = datasets.ImageFolder(train, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)

valset = datasets.ImageFolder(val, transform=transform_train)
val_loader = torch.utils.data.DataLoader(valset,
                                          batch_size=args.batch_size,
                                          shuffle=False, num_workers=0)

testset = datasets.ImageFolder(test, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)

n_classe = len(trainset.classes)
print("Class",n_classe)


print('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
net = models.resnet50(pretrained=True)

for param in net.parameters():
  param.requires_grad = False

net.fc = nn.Linear(net.fc.in_features, n_classe)

model = net.to(device)


if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = optim.SGD(model.parameters(), lr=0.0001 ,momentum=0.9) #weight_decay=5e-4
#optimizer_cnn = optim.Adam(net.parameters(), lr=0.0001)

"""
Training
"""
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_cnn.zero_grad()
        outputs = net(inputs)

        loss = criterion_cnn(outputs, targets)
        loss.backward()
        optimizer_cnn.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        writer.add_scalar('Training/ACC_',100.*correct/total, (epoch*len(train_loader.dataset)/args.batch_size)+batch_idx)
        writer.add_scalar('Training/loss_',train_loss/(batch_idx+1),(epoch*len(train_loader.dataset)/args.batch_size)+batch_idx)
    print('\n %d',correct/total*100)
    writer.add_scalar('Training/ACC',correct/total*100, epoch)


"""
val
"""


"""
Test
"""

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print("ACC_test",acc)
    return acc


"""
Feature extration
"""
def fatureEx_(data,model):
    model.eval()

    fe_ = []
    label_= []
    
    #for param in model_.parameters():
       # param.requires_grad = False

    model.fc = nn.Linear(2048, 2048)
    feature_extractor = model.to(device)

    print(feature_extractor)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_fe = feature_extractor(inputs)
            fe_.append(outputs_fe.data.cpu().numpy())
            label_.append(targets.data.cpu().numpy())
    
    return fe_, label_


"""
Utils
"""

def unmount_batch_v2(feature_t,true_l):
  feature_img_label = []
  feature = []
  true_label = []
  for i in range(len(feature_t)):
    for j in range(len(feature_t[i])):
      feature.append(feature_t[i][j])
      true_label.append(true_l[i][j])
  return np.array(feature),np.array(true_label)



def save_model(model):
    torch.save(model.state_dict(), "results/"+seed+"_model.pt")

"""
Main
"""

def main():
    print("test with pre treining on cifar10 or cifar 100")
    for epoch in range(0, args.epoch):
        train(epoch)
        acc = test()
        if(acc > best_acc):
            print("save modelo")
            save_model(model)



    model.load_state_dict(torch.load("results/"+seed+"_model.pt",map_location=device)) # carregar o modelo treinado "CNN"

    fe_train_s, label_train_s = fatureEx_(train_loader,model)
    feature_t, label_t = unmount_batch_v2(fe_train_s, label_train_s)
    np.savez("results/"+seed+'_train', feature_t,label_t)

    fe_test_s, label_test_s = fatureEx_(test_loader,model)
    feature_tt, label_tt = unmount_batch_v2(fe_test_s, label_test_s)
    np.savez("results/"+seed+'_test', feature_tt,label_tt)

    fe_val_s, label_val_s = fatureEx_(val_loader,model)
    feature_val, label_val = unmount_batch_v2(fe_val_s, label_val_s)
    print(feature_val.shape)
    np.savez("results/"+seed+'_val', feature_val,label_val)
 


if __name__ == '__main__':
    main()
