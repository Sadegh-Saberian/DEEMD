import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import h5py
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import h5py
from collections import deque


parser = argparse.ArgumentParser(description='MIL')
parser.add_argument('--k', type=int, default='1', help='k top patches to train on')
parser.add_argument('--i', type=int, default='1', help='experiment repetition number')
parser.add_argument('--train_lib', type=str, default='./meta/HRCE_train_metadata.csv', help='path to the metadata csv file for training')
parser.add_argument('--val_lib', type=str, default='./meta/HRCE_valid_metadata.csv', help='path to the metadata csv file for validation')
parser.add_argument('--untreated_test_lib', type=str, default='./meta/HRCE_train_metadata.csv', help='path to the metadata csv file for training')
parser.add_argument('--treated_test_lib', type=str, default='./meta/HRCE_valid_metadata.csv', help='path to the metadata csv file for validation')
parser.add_argument('--base_path', type=str, default='/project/compbio-lab/MIL-COVID19/RxRx19a', help='base path for reading the image files')
parser.add_argument('--batch_size', type=int, default='128', help='batch size for training and updating weights')
parser.add_argument('--n_workers', type=int, default='1', help='number of workers for dataloaders')
parser.add_argument('--overlap', type=float, default='0.5', help='patch overlapping factor')
parser.add_argument('--patch_size', type=int, default='256', help='size of the extracted patches')
parser.add_argument('--n_classes', type=int, default='2', help='number of the classes in the dataset')
parser.add_argument('--output_path', type=str, default='/project/compbio-lab/MIL-COVID19/RxRx19a/output/', help='base path for storing the model params')

global args
args = parser.parse_args()

Max_len_dq = 500
train_lib = args.train_lib
val_lib = args.val_lib
treated_test_lib = args.treated_test_lib
untreated_test_lib = args.untreated_test_lib
base_path = args.base_path
batch_size = args.batch_size
workers = args.n_workers
k_top = int(args.k)
rep_n = (args.i)
overlap = 1 - args.overlap
patch_size = args.patch_size
n_classes = args.n_classes
print(k_top)
best_acc = 0.0
output = args.output_path



class MILdataset_RxRx19(data.Dataset):
    def __init__(self, libraryfile='', base_path='', patch_size=256, overlap=0.75, transform=None):
        start_time = time.time()
        data_df = pd.read_csv(libraryfile)
        num_bags = data_df.shape[0]
        bagIDX_lst = []
        targets_lst = []
        treatment_lst = []
        instance_lvl_targets = []
        grid = []
        status = []
        condition_lst = []
        path_lst = []
        categories = ['background'] + list(np.unique(data_df.disease_condition))
        tmp_labels = pd.Categorical(data_df.disease_condition, categories=categories, ordered=True).codes.copy()
        ## Index(['background', 'Active SARS-CoV-2', 'Mock', 'UV Inactivated SARS-CoV-2'], dtype='object')
        ## Categories (4, object): [background < Active SARS-CoV-2 < Mock < UV Inactivated SARS-CoV-2]
        for i in range(len(tmp_labels)):
            if tmp_labels[i] == 1:
                continue
            elif tmp_labels[i] == 2 or tmp_labels[i] == 3:
                tmp_labels[i] = 0
        data_df['label'] = tmp_labels

        for index, row in data_df.iterrows():
            if index % 250 == 0:
                print(float(index / data_df.shape[0]))
            tmp_path = row['path']
            tmp_target = row['label']
            path_lst.append(base_path + tmp_path)
            ## load the images with loadImage method

            img_size_x = 1024
            img_size_y = 1024
            coordinates = [(i, j) for j in range(0, img_size_x - patch_size + 1, int(patch_size * (1 - overlap))) for i
                           in range(0, img_size_y - patch_size + 1, int(patch_size * (1 - overlap)))]

            grid.extend(coordinates)
            status.extend([0] * len(coordinates))
            bagIDX_lst.extend([index] * len(coordinates))
            instance_lvl_targets.extend([tmp_target] * len(coordinates))
            targets_lst.append(np.float32(tmp_target))
            treatment_lst.append(row['treatment'])
            condition_lst.append(row.disease_condition)

        self.treatment = treatment_lst
        self.condition = condition_lst
        self.targets = targets_lst
        self.grid = grid
        self.status = status
        self.bagIDX = bagIDX_lst
        self.transform = transform
        self.mode = None
        self.patch_size = patch_size
        self.t_data = []
        self.instance_lvl_targets = instance_lvl_targets
        self.paths = path_lst

        self.image_dq = deque(maxlen=Max_len_dq)
        self.bagIDX_dq = deque(maxlen=Max_len_dq)

        print('Number of bags: {}, Number of instances: {} , exec time: {}'.format(len(targets_lst), len(grid),
                                                                                   time.time() - start_time))

    def loadImage(self, idx):
        if idx in self.bagIDX_dq:
            img = self.image_dq[self.bagIDX_dq.index(idx)]
        else:
            tmp_path = self.paths[idx]
            for i in range(1, 6, 1):
                tmp_path_i = tmp_path.replace('*', str(i))
                tmp_img = Image.open(tmp_path_i)
                if i == 1:
                    img = tmp_img.copy()
                    continue
                img = np.dstack((img, tmp_img))
            self.bagIDX_dq.append(idx)
            self.image_dq.append(img)
        return img

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs_top, idxs_bot, idxs_reg):
        ## codes : 0 : kbot -> background
        ##         1 : reg  -> regulation
        ##         2 : ktop -> target
        self.status = [0.] * len(idxs_bot) + [1.] * len(idxs_reg) + [2.] * len(idxs_top)
        t_data_bot = [(self.bagIDX[x], self.grid[x], self.targets[self.bagIDX[x]]) for x in idxs_bot]
        t_data_reg = [(self.bagIDX[x], self.grid[x], self.targets[self.bagIDX[x]]) for x in idxs_reg]
        t_data_top = [(self.bagIDX[x], self.grid[x], self.targets[self.bagIDX[x]]) for x in idxs_top]

        t_data_list = t_data_bot + t_data_reg + t_data_top
        self.t_data = t_data_list

    def shuffletraindata(self):
        c = list(zip(self.t_data, self.status))
        random.shuffle(c)
        self.t_data, self.status = zip(*c)

    def __getitem__(self, index):
        if self.mode == 1:
            bagIDX = self.bagIDX[index]
            coord = self.grid[index]
            tmp_bag = self.loadImage(bagIDX)
            img = tmp_bag[coord[0]:(coord[0] + self.patch_size),
                  coord[1]:(coord[1] + self.patch_size), :]
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            bagIDX, grid, target = self.t_data[index]
            status = self.status[index]
            tmp_bag = self.loadImage(bagIDX)
            img = tmp_bag[grid[0]:(grid[0] + self.patch_size),
                  grid[1]:(grid[1] + self.patch_size), :]
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)


def inference(run, loader, model):
    start_time = time.time()
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    features = torch.FloatTensor(len(loader.dataset),512)
    with torch.no_grad():
        for i, input in enumerate(loader):
            if i % 250 == 0:
                print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}] , exec time: {}'.format(run+1, nepochs, i+1, len(loader) , time.time() - start_time))
                start_time = time.time()
            input = input.cuda()
            output,tmp_feature = model(input)
            output = F.softmax(output, dim=1)
            probs[i*batch_size:i*batch_size+input.size(0)] = output.detach()[:,1].clone()
            features[i*batch_size:i*batch_size+input.size(0),:] = tmp_feature.detach()[:,:].clone()
    features = features.cpu().numpy()
    return probs.cpu().numpy() , features


def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out


def group_kth_max(groups,data,k):
    grp_unq = np.unique(groups)
    out = np.zeros(len(grp_unq))
    for i,g in enumerate(grp_unq):
        sampled_idx = range(i*49,(i+1)*49)
        tmp_data = [data[x] for x in sampled_idx]
        out[i] = sorted(tmp_data)[-k]
    return out

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

# def visualize(loader, model):
#     model.eval()
#     features = torch.FloatTensor(len(loader.dataset),512)
#     with torch.no_grad():
#         for i, input in enumerate(loader):
#             if i % 250 ==0:
#                 print('inference progress:{}'.format(float(i*batch_size/len(loader.dataset))))
#             input = torch.FloatTensor(input.float()).cuda()
#             _,tmp_feature = model(input)
#             features[i*batch_size:i*batch_size+input.size(0),:] = tmp_feature.detach()[:,:].clone()
#     features = features.cpu().numpy()
#     return features

model = ResnetModel_mod(n_classes)
model.cuda()
pos_weight = torch.FloatTensor([1,0.9])
criterion = nn.CrossEntropyLoss(weight= pos_weight).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
cudnn.benchmark = True



checkpoint = torch.load(output+'SMIL_checkpoint_best_pre_'+str(k_top)+'_'+rep_n+'.pth')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']

model.eval()

normalize = transforms.Normalize(mean=[0.013 , 0.080 , 0.080 , 0.050 , 0.070],
                                 std=[0.018 , 0.056 , 0.060 , 0.038 , 0.040])
trans = transforms.Compose([transforms.ToTensor(), normalize])

train_dset = MILdataset_RxRx19(libraryfile=train_lib,patch_size = patch_size , base_path=base_path , transform = trans, overlap=overlap)
train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

val_dset = MILdataset_RxRx19(libraryfile=val_lib,patch_size = patch_size , base_path=base_path , transform = trans, overlap=overlap)
val_loader = torch.utils.data.DataLoader(
    val_dset,
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)

treated_test_dset = MILdataset_RxRx19(libraryfile=treated_test_lib,patch_size = patch_size , base_path=base_path , transform = trans, overlap=overlap)
treated_test_loader = torch.utils.data.DataLoader(
        treated_test_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

untreated_test_dset = MILdataset_RxRx19(libraryfile=untreated_test_lib,patch_size = patch_size , base_path=base_path , transform = trans, overlap=overlap)
untreated_test_loader = torch.utils.data.DataLoader(
        untreated_test_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


print(k_top)
train_dset.setmode(1)
print('working on train dset!')
train_probs,train_features = inference(epoch, train_loader, model)

val_dset.setmode(1)
print('working on val dset!')
val_probs,val_features = inference(epoch, val_loader, model)

untreated_test_dset.setmode(1)
print('working on untreated test dset!')
untreated_test_probs,untreated_test_features = inference(epoch, untreated_test_loader, model)

treated_test_dset.setmode(1)
print('working on treated test dset!')
treated_test_probs,treated_test_features = inference(epoch, treated_test_loader, model)


target_names = ['Mock','Active SARS-CoV-2']



print('working on train dset!')
h5f = h5py.File(output+'train_dset_'+str(k_top)+'_'+rep_n+'.h5', 'w')
h5f.create_dataset('probs', data=train_probs)
h5f.create_dataset('features', data=train_features)
h5f.close()
print('done train dset!')


print('working on val dset!')
h5f = h5py.File(output+'val_dset_'+str(k_top)+'_'+rep_n+'.h5', 'w')
h5f.create_dataset('probs', data=val_probs)
h5f.create_dataset('features', data=val_features)
h5f.close()
print('done val dset!')



print('working on treated test dset!')
h5f = h5py.File(output+'treated_test_dset_'+str(k_top)+'_'+rep_n+'.h5', 'w')
h5f.create_dataset('probs', data=treated_test_probs)
h5f.create_dataset('features', data=treated_test_features)
h5f.close()
print('done test dset!')



print('working on untreated test dset!')
h5f = h5py.File(output+'untreated_test_dset_'+str(k_top)+'_'+rep_n+'.h5', 'w')
h5f.create_dataset('probs', data=untreated_test_probs)
h5f.create_dataset('features', data=untreated_test_features)
h5f.close()
print('done test dset!')

maxs = group_kth_max(np.array(val_dset.bagIDX), val_probs, k_top)

print('train')
# maxs = group_max(np.array(train_dset.bagIDX), train_probs, len(train_dset.targets))
maxs = group_kth_max(np.array(train_dset.bagIDX), train_probs, k_top)
train_pred = [1 if x >= 0.5 else 0 for x in maxs]
cr = (classification_report(train_dset.targets, train_pred, target_names=target_names ))
print(cr)
y_actu = pd.Series(train_dset.targets, name='Actual')
y_pred = pd.Series(train_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)

print('val')
# maxs = group_max(np.array(val_dset.bagIDX), val_probs, len(val_dset.targets))
maxs = group_kth_max(np.array(val_dset.bagIDX), val_probs, k_top)
val_pred = [1 if x >= 0.5 else 0 for x in maxs]
cr = (classification_report(val_dset.targets, val_pred, target_names=target_names ))
print(cr)
y_actu = pd.Series(val_dset.targets, name='Actual')
y_pred = pd.Series(val_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)

print('treated test')
# maxs = group_max(np.array(treated_test_dset.bagIDX), treated_test_probs, len(treated_test_dset.targets))
maxs = group_kth_max(np.array(treated_test_dset.bagIDX), treated_test_probs, k_top)
treated_test_pred = [1 if x >= 0.5 else 0 for x in maxs]
cr = (classification_report(treated_test_dset.targets, treated_test_pred, target_names=target_names ))
print(cr)
y_actu = pd.Series(treated_test_dset.targets, name='Actual')
y_pred = pd.Series(treated_test_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)

print('untreated test')
# maxs = group_max(np.array(untreated_test_dset.bagIDX), untreated_test_probs, len(untreated_test_dset.targets))
maxs = group_kth_max(np.array(untreated_test_dset.bagIDX), untreated_test_probs, k_top)

untreated_test_pred = [1 if x >= 0.5 else 0 for x in maxs]
cr = (classification_report(untreated_test_dset.targets, untreated_test_pred, target_names=target_names ))
print(cr)
y_actu = pd.Series(untreated_test_dset.targets, name='Actual')
y_pred = pd.Series(untreated_test_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)
