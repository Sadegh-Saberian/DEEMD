
import torch; import torchvision; print("Will print True below if success!"); print(torch.cuda.is_available());
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
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
import cv2
from torch.optim import lr_scheduler
import h5py


parser = argparse.ArgumentParser(description='MIL')
parser.add_argument('--k', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--i', type=str, default='', help='repetition number')
global args
args = parser.parse_args()
nepochs = 150

train_lib = './meta/HRCE_train_metadata.csv'
val_lib = './meta/HRCE_valid_metadata.csv'
treated_test_lib = './meta/HRCE_test_treated_metadata.csv'
untreated_test_lib = './meta/HRCE_test_untreated_metadata.csv'

base_path = '/project/compbio-lab/MIL-COVID19/RxRx19a'
test_every = 5
batch_size = 128
workers = 1
k_top = int(args.k)
rep_n = (args.i)
overlap = 1 - 0.5
patch_size = int(256)
n_classes = 2
print(k_top)
best_acc = 0.0
output = '/project/compbio-lab/MIL-COVID19/RxRx19a/output/'

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr
def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_kth_max(groups,data,k):
    grp_unq = np.unique(groups)
    out = np.zeros(len(grp_unq))
    for i,g in enumerate(grp_unq):
        sampled_idx = range(i*49,(i+1)*49)
        tmp_data = [data[x] for x in sampled_idx]
        out[i] = sorted(tmp_data)[-k]
    return out

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


class MILdataset_RxRx19(data.Dataset):
    def __init__(self, libraryfile='', base_path='', patch_size=256, overlap=0.75, transform=None):
        start_time = time.time()
        data_df = pd.read_csv(libraryfile)
        num_bags = data_df.shape[0]
        bags_lst = []
        bagIDX_lst = []
        targets_lst = []
        instances_lst = []
        bag_names_lst = []
        treatment_lst = []
        conc_lst = []
        instance_lvl_targets = []
        grid = []
        status = []
        condition_lst = []
        categories = ['background'] + list(np.unique(data_df.disease_condition))
        #         data_df['label'] = pd.Categorical(data_df.disease_condition,categories=categories,ordered = True).codes
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
            for i in range(1, 6, 1):
                tmp_path_i = base_path + tmp_path.replace('*', str(i))
                tmp_img = Image.open(tmp_path_i)
                if i == 1:
                    img = tmp_img.copy()
                    continue
                img = np.dstack((img, tmp_img))
            tmp_bag = img[:, :, :]

            img_size_x = img.shape[0]
            img_size_y = img.shape[1]
            coordinates = [(i, j) for j in range(0, img_size_x - patch_size + 1, int(patch_size * (1 - overlap))) for i
                           in range(0, img_size_y - patch_size + 1, int(patch_size * (1 - overlap)))]
            #             print('done creating the grid for image.')

            grid.extend(coordinates)
            status.extend([0] * len(coordinates))
            bags_lst.append(tmp_bag)
            bagIDX_lst.extend([index] * len(coordinates))
            instance_lvl_targets.extend([tmp_target] * len(coordinates))
            targets_lst.append(np.float32(tmp_target))
            treatment_lst.append(row['treatment'])
            conc_lst.append(row['treatment_conc'])
            condition_lst.append(row.disease_condition)
        #             print('done adding bag to the trainig lists.')
        #         self.weights = float(targets_lst.count(1)/(targets_lst.count(1) + targets_lst.count(0)))
        self.treatment = treatment_lst
        self.condition = condition_lst
        self.treatment_conc = conc_lst
        self.bags = bags_lst
        self.targets = targets_lst
        self.grid = grid
        self.status = status
        #         self.instances = instances_lst
        self.bagIDX = bagIDX_lst
        self.transform = transform
        self.mode = None
        #         self.feature_size = feature_size
        self.patch_size = patch_size
        self.t_data = []
        #         self.backgorund_class = background_class
        self.instance_lvl_targets = instance_lvl_targets
        print('Number of bags: {}, Number of instances: {} , exec time: {}'.format(len(bags_lst), len(grid),
                                                                                   time.time() - start_time))

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
            img = self.bags[bagIDX][coord[0]:(coord[0] + self.patch_size),
                  coord[1]:(coord[1] + self.patch_size), :]
            if self.transform is not None:
                #                 print(img.shape)
                img = self.transform(img)
            #                 print(img.shape)
            return img
        elif self.mode == 2:
            bagIDX, grid, target = self.t_data[index]
            status = self.status[index]
            img = self.bags[bagIDX][grid[0]:(grid[0] + self.patch_size),
                  grid[1]:(grid[1] + self.patch_size), :]
            if self.transform is not None:
                #                 print(img.shape)
                img = self.transform(img)
            #                 print(img.shape)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)


def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    features = torch.FloatTensor(len(loader.dataset), 512)
    with torch.no_grad():
        for i, input in enumerate(loader):
            if i % 250 == 0:
                print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run + 1, nepochs, i + 1, len(loader)))
            input = input.cuda()
            output, tmp_feature = model(input)
            output = F.softmax(output, dim=1)
            probs[i * batch_size:i * batch_size + input.size(0)] = output.detach()[:, 1].clone()
            features[i * batch_size:i * batch_size + input.size(0), :] = tmp_feature.detach()[:, :].clone()
    features = features.cpu().numpy()
    return probs.cpu().numpy(),features


def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        if i % 250 == 0:
            print('Training\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run + 1, nepochs, i + 1, len(loader)))
        #         input = input.cuda()
        input = torch.FloatTensor(input.float()).cuda()
        #         target = target.cuda()
        target = (target.to(torch.int64)).cuda()
        output, _ = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input.size(0)
    return running_loss / len(loader.dataset)

normalize = transforms.Normalize(mean=[0.013 , 0.080 , 0.080 , 0.050 , 0.070],
                                 std=[0.018 , 0.056 , 0.060 , 0.038 , 0.040])
trans = transforms.Compose([transforms.ToTensor(), normalize])

train_dset = MILdataset_RxRx19(libraryfile=train_lib,patch_size = patch_size , base_path=base_path , transform = trans, overlap=overlap)
train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
if val_lib:
    val_dset = MILdataset_RxRx19(libraryfile=val_lib,patch_size = patch_size , base_path=base_path , transform = trans, overlap=overlap)
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
pos_weight = torch.FloatTensor([1,1])
criterion = nn.CrossEntropyLoss(weight= pos_weight).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
cudnn.benchmark = True
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#open output file
fconv = open(os.path.join(output,'convergence'+str(k_top)+'_'+rep_n+'.csv'), 'w')
fconv.write('epoch,metric,value\n')
fconv.close()

#loop throuh epochs
for epoch in range(nepochs):
    train_dset.setmode(1)
    train_probs,train_features = inference(epoch, train_loader, model)
    topk = group_argtopk(np.array(train_dset.bagIDX), train_probs, k_top)
    train_dset.maketraindata(topk,[],[])
    train_dset.shuffletraindata()
    train_dset.setmode(2)
    loss = train(epoch, train_loader, model, criterion, optimizer)
    print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, nepochs, loss))
    fconv = open(os.path.join(output, 'convergence'+str(k_top)+'_'+rep_n+'.csv'), 'a')
    fconv.write('{},loss,{}\n'.format(epoch+1,loss))
    fconv.close()

    #Validation
    if val_lib and (epoch+1) % test_every == 0:
        val_dset.setmode(1)
        val_probs,val_features = inference(epoch, val_loader, model)
        # maxs = group_max(np.array(val_dset.bagIDX), val_probs, len(val_dset.targets))
        maxs = group_kth_max(np.array(val_dset.bagIDX), val_probs, k_top)
        val_pred = [1 if x >= 0.5 else 0 for x in maxs]
        err,fpr,fnr = calc_err(val_pred, val_dset.targets)
        print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, nepochs, err, fpr, fnr))
        fconv = open(os.path.join(output, 'convergence'+str(k_top)+'_'+rep_n+'.csv'), 'a')
        fconv.write('{},error,{}\n'.format(epoch+1, err))
        fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
        fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
        fconv.close()
        #Save best model
        err = (fpr+fnr)/2.
        if 1-err >= best_acc:
            best_acc = 1-err
            obj = {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()
            }
            torch.save(obj, os.path.join(output,'SMIL_checkpoint_best_pre_'+str(k_top)+'_'+rep_n+'.pth'))



#print('working on saving train dset!')
#h5f = h5py.File('train_dset_pred.h5', 'w')
#h5f.create_dataset('probs', data=train_probs)
#h5f.create_dataset('features' , data = train_features)
#h5f.close()
#print('done train dset!')


#print('working on saving val dset!')
#h5f = h5py.File('train_val_pred.h5', 'w')
#h5f.create_dataset('probs', data=val_probs)
#h5f.create_dataset('features' , data = val_features)
#h5f.close()
#print('done train dset!')


