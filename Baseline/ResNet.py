import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import sys
import os
import numpy as np
import argparse
import random
import PIL.Image as Image
import cv2



parser = argparse.ArgumentParser(description='MIL')
parser.add_argument('--test', type=int, default='5', help='the number of epochs between each validation procedure')
parser.add_argument('--e', type=int, default='100', help='number of epochs for training')
parser.add_argument('--train_lib', type=str, default='./meta/HRCE_train_metadata.csv', help='path to the metadata csv file for training')
parser.add_argument('--val_lib', type=str, default='./meta/HRCE_valid_metadata.csv', help='path to the metadata csv file for validation')
parser.add_argument('--base_path', type=str, default='/project/compbio-lab/MIL-COVID19/RxRx19a', help='base path for reading the image files')
parser.add_argument('--batch_size', type=int, default='16', help='batch size for training and updating weights')
parser.add_argument('--n_workers', type=int, default='1', help='number of workers for dataloaders')
parser.add_argument('--n_classes', type=int, default='2', help='number of the classes in the dataset')
parser.add_argument('--output_path', type=str, default='/project/compbio-lab/MIL-COVID19/RxRx19a/output/', help='base path for storing the model params')

global args
args = parser.parse_args()
nepochs = args.e
train_lib = args.train_lib
val_lib = args.val_lib
test_every = args.test
base_path = args.base_path
batch_size = args.batch_size
workers = args.n_workers
n_classes = args.n_classes
best_score = 0.0
output = args.output_path



class RxRx19_Classification_DL(data.Dataset):
    def __init__(self, libraryfile='', base_path='', transform=None):
        start_time = time.time()
        data_df = pd.read_csv(libraryfile)
        num_samples = data_df.shape[0]
        print(num_samples)
        data_df.reset_index(drop=True, inplace=True)

        samples_lst = []
        sampleIDX_lst = []
        targets_lst = []
        treatment_lst = []
        condition_lst = []

        data_df['label'] = data_df['disease_condition']
        label = {'Active SARS-CoV-2': 1, 'Mock': 0, 'UV Inactivated SARS-CoV-2': 0}
        data_df.label = [label[item] for item in data_df.label]

        for index, row in data_df.iterrows():
            if index % 250 == 0:
                print(float(index / data_df.shape[0]))
            tmp_path = row['path']
            tmp_target = row['label']
            path = base_path + tmp_path
            for i in range(1, 6, 1):
                tmp_path_i = base_path + tmp_path.replace('*', str(i))
                tmp_img = Image.open(tmp_path_i)
#                tmp_img = tmp_img.resize((224, 224), Image.BILINEAR)
                if i == 1:
                    img = tmp_img.copy()
                    continue
                img = np.dstack((img, tmp_img))
            tmp_bag = img[:, :, :]

            samples_lst.append(tmp_bag)
            sampleIDX_lst.append(index)
            targets_lst.append(tmp_target)
            treatment_lst.append(row['treatment'])
            condition_lst.append(row.disease_condition)

        self.treatment = treatment_lst
        self.condition = condition_lst
        self.samples = samples_lst
        self.targets = targets_lst
        self.sampleIDX = sampleIDX_lst
        self.transform = transform
        self.mode = None
        self.t_data = [(self.sampleIDX[x], self.targets[self.sampleIDX[x]]) for x in range(len(self.samples))]
        self.t_data = random.sample(self.t_data, len(self.t_data))

        ttarget = []
        for x in self.t_data:
            ttarget.append(x[1])
        self.ttarget = ttarget

        print('Number of samples: {}, exec time: {}'.format(len(samples_lst), time.time() - start_time))

    def __getitem__(self, index):
        sampleIDX, target = self.t_data[index]
        img = self.samples[sampleIDX]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.t_data)


def inference(run, loader, model, n_classes):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset), n_classes)
    features = torch.FloatTensor(len(loader.dataset), 512)
    with torch.no_grad():
        for i, (inputs, target) in enumerate(loader):
            if i % 250 == 0:
                print('inference progress:{}'.format(float(i * batch_size / len(loader.dataset))))
            inputs = torch.FloatTensor(inputs.float()).cuda()
            output, tmp_feature = model(inputs)
            output = F.softmax(output, dim=1)
            features[i * batch_size:i * batch_size + inputs.size(0), :] = tmp_feature.detach()[:, :].clone()
            probs[i * batch_size:i * batch_size + inputs.size(0), :] = output.detach()[:, :].clone()
    probs = probs.cpu().numpy()
    features = features.cpu().numpy()
    return probs,features


def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (inputs, target) in enumerate(loader):
        if i % 250 == 0:
            print('training progress:{}'.format(float(i * batch_size / len(loader.dataset))))
        inputs = torch.FloatTensor(inputs.float()).cuda()
        target = (target.to(torch.int64)).cuda()
        optimizer.zero_grad()
        output, _  = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


# normalize = transforms.Normalize(mean=(0.5,0.5,0.5,0.5,0.5),std=(0.1,0.1,0.1,0.1,0.1))
normalize = transforms.Normalize(mean=[0.013 , 0.080 , 0.080 , 0.050 , 0.070],
                                 std=[0.018 , 0.056 , 0.060 , 0.038 , 0.040])
trans = transforms.Compose([transforms.ToTensor(), normalize])

train_dset = RxRx19_Classification_DL(libraryfile=train_lib, base_path=base_path, transform = trans)

train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

val_dset = RxRx19_Classification_DL(libraryfile=val_lib, base_path=base_path, transform = trans)

val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


class ResnetModel_mod(nn.Module):
    def __init__(self, n_classes):
        super(ResnetModel_mod, self).__init__()

        self.model_resnet = models.resnet34(True)
        self.model_resnet.conv1 = torch.nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, n_classes, bias=True)

    def forward(self, x):
        x = self.model_resnet(x)
        prediction = self.fc1(x)
        features = self.model_resnet.fc(x)
        return prediction, features


model = ResnetModel_mod(n_classes)
model.cuda()
w = torch.FloatTensor([0.5,0.5])
criterion = nn.CrossEntropyLoss(w).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
cudnn.benchmark = True
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

train_accu_list = []
val_accu_list = []
train_lost_list = []
valscore_list = []
for epoch in range(nepochs):
    loss = train(epoch, train_loader, model, criterion, optimizer)
    train_lost_list.append(loss)
    exp_lr_scheduler.step()
    print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch + 1, nepochs, loss))

    train_probs, _ = inference(epoch, train_loader, model, n_classes)
    train_max_c_probs = np.argmax(train_probs, axis=1)

    train_pred_lst = [float(c) for c in train_max_c_probs]

    target_names = ['Infected', 'Healthy']
    label_names = ['Infected', 'Healthy']
    cr = (classification_report(train_dset.ttarget, train_pred_lst, output_dict=True, target_names=target_names,
                                labels=range(2)))
    print(classification_report(train_dset.ttarget, train_pred_lst, target_names=target_names, labels=range(2)))

    y_actu = pd.Series(train_dset.ttarget, name='Actual')
    y_pred = pd.Series(train_pred_lst, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)

    accu = list(y_actu == y_pred)
    accuracy = sum(accu) / len(accu)
    train_accu_list.append(accuracy)

    print('The training accuracy is', accuracy)
    print('Training confusion')
    print(df_confusion)
    print('#' * 20)
    if (epoch + 1) % test_every == 0:
        val_probs, _ = inference(epoch, val_loader, model, n_classes)
        max_c_probs = np.argmax(val_probs, axis=1)

        pred_lst = [float(c) for c in max_c_probs]
        target_names = ['Infected', 'Healthy']
        label_names = ['Infected', 'Healthy']
        cr = (classification_report(val_dset.ttarget, pred_lst, output_dict=True, target_names=target_names,
                                    labels=range(2)))
        print(classification_report(val_dset.ttarget, pred_lst, target_names=target_names, labels=range(2)))

        val_score = cr['weighted avg']['f1-score']
        valscore_list.append(val_score)

        print('val_score:{}'.format(val_score))
        y_actu = pd.Series(val_dset.ttarget, name='Actual')
        y_pred = pd.Series(pred_lst, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        accu_val = list(y_actu == y_pred)
        accuracy_val = sum(accu_val) / len(accu_val)
        val_accu_list.append(accuracy_val)
        print('The testing accuracy is', accuracy_val)
        print('Testing confusion')
        print(df_confusion)

        if val_score >= best_score:
            best_score = val_score
            obj = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_score,
                'optimizer': optimizer.state_dict()
            }
            torch.save(obj, os.path.join('./', 'classification_checkpoint_best_f.pth'))
