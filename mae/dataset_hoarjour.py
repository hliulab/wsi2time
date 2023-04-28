import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import h5py
import gc

class Whole_Slide_Bag_Tile(Dataset):
    def __init__(self, data_dir, slide_name):
        self.data_dir = data_dir
        self.slide_name = slide_name.split(".svs")[0]

    def __getitem__(self, idx):
        q_path = os.path.join(self.data_dir, "tiles", f"{self.slide_name}_{idx}_q.png")
        k_path = os.path.join(self.data_dir, "tiles", f"{self.slide_name}_{idx}_k.png")

        if not os.path.isfile(q_path) and os.path.isfile(k_path):
            q_path = k_path
        elif os.path.isfile(q_path) and not os.path.isfile(k_path):
            k_path = q_path
        elif not os.path.isfile(q_path) and not os.path.isfile(k_path):
            raise Exception("数据出问题")
        # print(f"q_path   {q_path}")
        # print(f"k_path   {k_path}")

        q = Image.open(q_path).convert("RGB")
        try:
            k = Image.open(k_path).convert("RGB")
        except Exception:
            k = Image.open(q_path).convert("RGB")

        q = transforms.ToTensor()(q)
        k = transforms.ToTensor()(k)

        return [q, k], 1

    def __len__(self):
        return len(os.listdir(self.data_dir + "/tiles")) // 2


class Gene_Reg_Dataset(Dataset):
    def __init__(self, h5_dir, csv_path, test=False):
        self.h5_dir = h5_dir
        self.csv_path = csv_path
        self.train_data = pd.read_csv(csv_path, index_col="slide_id")
        self.test = test

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        h5_file = self.train_data.index[idx]
        h5_file = h5_file.split(".h5")[0] + ".h5"
        h5_path = os.path.join(self.h5_dir, h5_file)

        # 读取h5格式的特征文件
        f = h5py.File(h5_path, 'r')
        features = f["features"][()]
        f.close()

        features_ = torch.from_numpy(features)

        # 读取label
        labels = self.train_data.iloc[idx].values
        labels = np.expand_dims(labels, axis=0)
        labels = labels.astype(np.float32)
        labels = torch.from_numpy(labels)
        if self.test:
            return features_, h5_file
        else:
            return features_, labels


class Simple_Dataset(Dataset):

    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        self.feats = os.listdir(feat_dir)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        # 读取h5格式的特征文件
        h5_name = self.feats[idx]
        h5_path = os.path.join(self.feat_dir, h5_name)
        f = h5py.File(h5_path, 'r')
        features = f["features"][()]
        f.close()

        features_ = torch.from_numpy(features)

        return features_, h5_name


class Multi_Gene_Reg_Dataset(Dataset):

    def __init__(self, h5_dir, csv_path, test=False, norm=True):
        self.h5_dir = h5_dir
        self.csv_path = csv_path
        self.train_data = pd.read_csv(csv_path)
        self.test = test
        self.gene_name = None

        self.features = []

        # 一口气读出全部的h5文件
        for slide_id in self.train_data["slide_id"].values:
            h5_file = slide_id.split(".svs")[0] + ".h5"
            h5_path = os.path.join(self.h5_dir, h5_file)

            # 读取h5格式的特征文件
            f = h5py.File(h5_path, 'r')
            features = f["features"][()]
            f.close()
            features = torch.from_numpy(features)

            self.features.append(features)
        print(f"全部h5特征文件读取完毕,长度为:{len(self.features)}")

        # 对基因数据进行log10(1 + n)，归一化
        if norm:
            self._norm()

    def switch(self, gene_name):
        self.gene_name = gene_name

    def get_all_gene_names(self):
        return self.train_data.columns[1:]

    def __len__(self):
        return self.train_data.shape[0]

    def _norm(self):
        columns = self.train_data.columns[1:]
        # log10(n + 1)
        for col in columns.values:
            self.train_data[col] = np.log10(self.train_data[col].values + 1)
        print(f"{len(columns.values)}个基因数据归一化完毕")

    def __getitem__(self, idx):
        if idx == 0:
            if self.test:
                print(f"使用基因数据 {self.gene_name} 进行test")
            else:
                print(f"使用基因数据 {self.gene_name} 进行train")
        labels = self.train_data[self.gene_name][idx]
        labels = np.expand_dims(labels, axis=0)
        labels = labels.astype(np.float32)
        labels = torch.from_numpy(labels)
        return self.features[idx], labels


class Multi_Gene_Reg_Dataset_16(Dataset):

    def __init__(self, h5_dir, slide_path, gene_data_path, test=False, norm=True):
        self.h5_dir = h5_dir
        self.slide_path = slide_path
        self.gene_data_path = gene_data_path

        self.slide_data = pd.read_csv(slide_path)
        self.gene_data = pd.read_csv(gene_data_path)

        self.test = test
        self.gene_name = None

        self.features = []

        # 一口气读出全部的h5文件
        for slide_id in self.slide_data["slide_id"].values:
            h5_file = slide_id.split(".svs")[0] + ".h5"
            h5_path = os.path.join(self.h5_dir, h5_file)

            # 读取h5格式的特征文件
            f = h5py.File(h5_path, 'r')
            features = f["features"][()]
            f.close()
            features = torch.from_numpy(features)

            self.features.append(features)
        print(f"全部h5特征文件读取完毕,长度为:{len(self.features)}")

        # 对基因数据进行log10(1 + n)，归一化
        if norm:
            self._norm()
        self.norm_gene_data = self.gene_data.copy(deep=True)
        self.norm_gene_data.set_index("slide_id", inplace=True)

    def switch(self, gene_name):
        self.gene_name = gene_name

    def get_all_gene_names(self):
        return self.gene_data.columns[1:]

    def __len__(self):
        return self.slide_data.shape[0]

    def _norm(self):
        columns = self.gene_data.columns[1:]
        # log10(n + 1)
        for col in columns.values:
            self.gene_data[col] = np.log10(self.gene_data[col].values + 1)
        print(f"{len(columns.values)}个基因数据归一化完毕")

    def __getitem__(self, idx):
        if idx == 0:
            if self.test:
                print(f"使用基因数据 {self.gene_name} 进行test")
            else:
                print(f"使用基因数据 {self.gene_name} 进行train")
        labels = self.norm_gene_data.loc[self.slide_data["slide_id"][idx], self.gene_name]
        labels = np.expand_dims(labels, axis=0)
        labels = labels.astype(np.float32)
        labels = torch.from_numpy(labels)
        return self.features[idx], labels
