import os

import torch
from torch.utils.data import Dataset
import h5py
import pandas as pd

# from train_my_attention import CLAM_SB

#root_gen = r'/data/user_zy/project3/pytorch/clam/FEATURES_DIRECTORY/h5_files/'
#root_gen = r'F:\PycharmProjects\AI\WSI_Linux\PATCHES/'
root_gen =r'/data/wsi/tcga/breast_ts_feats/h5_files/ADCO_tcga_new/'

# 返回一个1×1000的numpy数组
def read_h5file(path):
    with h5py.File(path, 'r') as hdf5_file:
        features = hdf5_file['features'][:]
    return features


class FeatureDataset(Dataset):
    def __init__(self, feature_path=None, data_path=None, dataset_type=None,dateset=None):
        super(FeatureDataset, self).__init__()
        self.path = feature_path
        self.data_path = data_path
        self.feature = []

        df = pd.read_csv(data_path, sep=',', header=0)
        if dataset_type == 'train':
            for idx, row in df.iterrows():  # h5文件的路径
                y = list(row)
                self.feature.append((root_gen + str(y[0]), y[1:1+dateset]))
                if idx == 1300:
                    break

        elif dataset_type == 'valid':
            for idx, row in df.iterrows():  # h5文件的路径
                if 1300 <= idx <= 1600:
                    y = list(row)
                    self.feature.append((root_gen + str(y[0]), y[1:1+dateset]))


    def __getitem__(self, item) -> tuple:

        feature_h5path, data = self.feature[item]
        with h5py.File(feature_h5path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
        features = torch.from_numpy(features)
        data = torch.as_tensor(data)
        return features, data

    def __len__(self) -> int:
        return len(self.feature)

class FeatureDataset_mo(Dataset):
    def __init__(self, feature_path=None, data_path=None, dataset_type=None, is_normalized=False, is_mean=False,k=None,
                 is_exp=False, is_max=False):
        super(FeatureDataset_mo, self).__init__()
        self.path = feature_path
        self.data_path = data_path
        self.is_normalized = is_normalized
        self.is_mean = is_mean
        self.is_max = is_max
        self.is_exp = is_exp
        self.feature = []
        self.k = k
        df = pd.read_csv(data_path, sep=',', header=0)

        colums_list = df.columns.tolist()
        train_boundary=int(0.2* len(df))
        val_boundary=len(df)
        # for idx, row in df.iterrows():
        #     y = row['path']
        #     self.feature.append((os.path.join(self.path, str(y)),list(row[colums_list.index('B.cells.naive'):colums_list.index('Neutrophils') + 1])))
        if dataset_type == 'train':
            for idx, row in df.iterrows():
                if idx <k*train_boundary or val_boundary>=idx>(k+1)*train_boundary:

                    y = row['path']# h5文件的路径
                    # print(all_protein.index('X1433EPSILON'), all_protein.index('ANNEXINVII'))
                    self.feature.append((os.path.join(self.path , str(y)),
                         list(row[colums_list.index('B.cells.naive'):colums_list.index('Neutrophils') + 1])))


        elif dataset_type == 'valid':
            for idx, row in df.iterrows():  # h5文件的路径
                if idx >=k*train_boundary and idx <=(k+1)*train_boundary:
                    y = row['path']
                    self.feature.append((os.path.join(self.path , str(y)),
                                         list(row[colums_list.index('B.cells.naive'):colums_list.index('Neutrophils') + 1])))
                if idx == val_boundary:
                    break
        # else:
        #     for idx, row in df.iterrows():
        #         if idx >= val_boundary:
        #             y = row['path']
        #             self.feature.append((os.path.join(self.path, str(y)),
        #                                  list(row[colums_list.index('MKI67'):colums_list.index('TFRC') + 1])))

    def __getitem__(self, item) -> tuple:
        feature_h5path, data = self.feature[item]
        with h5py.File(feature_h5path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            hdf5_file.close()
        features = torch.from_numpy(features)
        if self.is_mean:
            features = torch.mean(features, dim=0)
        elif self.is_max:
            features, _ = torch.max(features, dim=0)
        if self.is_normalized:  # 是否归一化
            data = torch.Tensor(normalized(data))
            #features = torch.Tensor(features)
        elif self.is_exp:
            data = torch.Tensor(data)
            data = torch.exp(data)
        else:
            data = torch.as_tensor(data)
            #data = torch.log10(4 + data)
        return features, data

    def __len__(self) -> int:
        return len(self.feature)

class FeatureDataset_gene(Dataset):
    def __init__(self, feature_path=None, data_path=None, dataset_type=None, is_normalized=False, is_mean=False,k=None,
                 is_exp=False, is_max=False):
        super(FeatureDataset_gene, self).__init__()
        self.path = feature_path
        self.data_path = data_path
        self.is_normalized = is_normalized
        self.is_mean = is_mean
        self.is_max = is_max
        self.is_exp = is_exp
        self.feature = []
        self.k = k
        df = pd.read_csv(data_path, sep=',', header=0)

        colums_list = df.columns.tolist()
        train_boundary=int(0.2* len(df))
        val_boundary=len(df)
        # for idx, row in df.iterrows():
        #     y = row['path']
        #     self.feature.append((os.path.join(self.path, str(y)),list(row[colums_list.index('B.cells.naive'):colums_list.index('Neutrophils') + 1])))
        if dataset_type == 'train':
            for idx, row in df.iterrows():
                if idx <k*train_boundary or val_boundary>=idx>(k+1)*train_boundary:

                    y = row['path']# h5文件的路径
                    # print(all_protein.index('X1433EPSILON'), all_protein.index('ANNEXINVII'))
                    self.feature.append((os.path.join(self.path , str(y)),
                         list(row[colums_list.index('zcor'):colums_list.index('fcor') + 1])))


        elif dataset_type == 'valid':
            for idx, row in df.iterrows():  # h5文件的路径
                if idx >=k*train_boundary and idx <=(k+1)*train_boundary:
                    y = row['path']
                    self.feature.append((os.path.join(self.path , str(y)),
                                         list(row[colums_list.index('zcor'):colums_list.index('fcor') + 1])))
                if idx == val_boundary:
                    break
        # else:
        #     for idx, row in df.iterrows():
        #         if idx >= val_boundary:
        #             y = row['path']
        #             self.feature.append((os.path.join(self.path, str(y)),
        #                                  list(row[colums_list.index('MKI67'):colums_list.index('TFRC') + 1])))

    def __getitem__(self, item) -> tuple:
        feature_h5path, data = self.feature[item]
        with h5py.File(feature_h5path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            hdf5_file.close()
        features = torch.from_numpy(features)
        if self.is_mean:
            features = torch.mean(features, dim=0)
        elif self.is_max:
            features, _ = torch.max(features, dim=0)
        if self.is_normalized:  # 是否归一化
            data = torch.Tensor(normalized(data))
            #features = torch.Tensor(features)
        elif self.is_exp:
            data = torch.Tensor(data)
            data = torch.exp(data)
        else:
            data = torch.as_tensor(data)
            #data = torch.log10(4 + data)
        return features, data

    def __len__(self) -> int:
        return len(self.feature)
if __name__ == '__main__':
    my = FeatureDataset(feature_path=r'F:\XXD\PycharmProjects\WSI\PATCHES', dataset_type='train')
    you = FeatureDataset(feature_path=r'F:\XXD\PycharmProjects\WSI\PATCHES', dataset_type='valid')
    print(my.__len__())
    print(you.__len__())
    # x = read_h5file(
    #     r'F:\XXD\PycharmProjects\WSI\PATCHES\TCGA-05-4397-01A-01-BS1.21dd4531-70cf-47d6-955b-66af4ea85faf.h5')
    # print(type(x))
    # print(x.shape)
    # x = torch.from_numpy(x)
    # clam = CLAM_SB()
    # print(clam(x))
