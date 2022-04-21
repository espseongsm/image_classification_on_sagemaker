
import argparse
import os
import matplotlib.pyplot as plt
import time
import os
import copy
from datetime import datetime
from pytz import timezone
from tqdm import tqdm
import zipfile
import shutil
from pathlib import Path
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models

parser = argparse.ArgumentParser()

# 하이퍼파라미터 설정
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4)

# SageMaker Container 환경 설정
parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])


args, _ = parser.parse_known_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization)
# 검증을 위한 일반화
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([224, 224]),
#         transforms.RandomRotation(degrees=(0, 180)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = args.data
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        start_time = time.time()
        
        start = datetime.now(timezone('Asia/Seoul')
                            ).strftime('%Y-%m-%d %H:%M:%S')
        print('Start = {}'.format(start))

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]*100

            print('{:10}: Loss - {:10.4f} | Acc - {:10.2f}%'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 deep copy함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        finish = datetime.now(timezone('Asia/Seoul')
                            ).strftime('%Y-%m-%d %H:%M:%S')
        print('Finish = {}'.format(finish))
        
        time_elapsed = time.time() - start_time
        print('Time: {:10.2f}m'.format(time_elapsed/60))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:10.0f}hr {:10.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600)/60))
    print('Best val Acc: {:10.2f}%'.format(best_acc))

    # 가장 나은 모델 가중치를 불러와 저장함
    torch.save(best_model_wts, os.path.join(args.model_dir, 'model.pth'))
    logger.info("Model successfully saved at: {}".format(args.model_dir)) 


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# 모든 매개변수들이 최적화되었는지 관찰
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-1, momentum=0.9)

# 7에폭마다 0.1씩 학습률 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7,
                                       gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft,
                       exp_lr_scheduler,
                       num_epochs=args.num_epochs)
