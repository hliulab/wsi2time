import os
import umap.plot
import umap
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os
import argparse
from tqdm.asyncio import tqdm
from tqdm import tqdm

from datasets.dataset_generic import Generic_MIL_Dataset
from models.att_model import CLAM_SB_Class
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# optimizer.load_state_dict(checkpoint['optimizer'])
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--feature_path', default="../features'", type=str,
                    metavar='DIR',help='path to feature')
parser.add_argument('--model_path', type=str, default="../MODELS_result/adco_best_brca_sym.pth.tar")
parser.add_argument('--csv_path', type=str, default='../dataset_csv/sample_data.csv')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch_size')
args = parser.parse_args()
device = torch.device("cuda")

model = CLAM_SB_Class(n_classes=2, dropout=True)
checkpoint=torch.load(args.model_path,map_location=device)
model.load_state_dict(checkpoint['model'])
# torch.load('modelparameters.pth', map_location=lambda storage, loc: storage.cuda(1))
optimizer =optim.SGD(model.parameters(), lr=0.0003, momentum=0.9, weight_decay=1e-5)
train_dataset = Generic_MIL_Dataset(csv_path=args.csv_path,
                                            use_h5=args.use_h5,
                                            extract_model=args.extract_model,
                                            data_dir=args.data_root_dir,
                                            shuffle=False,
                                            seed=args.seed,
                                            print_info=True,
                                            label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
                                            patient_strat=False,
                                            ignore=[]
                                            )
val_dataset = Generic_MIL_Dataset(csv_path=args.csv_path,
                                  use_h5=args.use_h5,
                                  extract_model=args.extract_model,
                                  data_dir=args.data_root_dir,
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
                                  patient_strat=False,
                                  ignore=[]
                                  )
test_dataset = Generic_MIL_Dataset(csv_path=args.csv_path,
                                  use_h5=args.use_h5,
                                  extract_model=args.extract_model,
                                  data_dir=args.data_root_dir,
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
                                  patient_strat=False,
                                  ignore=[]
                                  )
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
model.eval
x1=np.zeros([1,512])
x2=np.zeros([1,512])
x3=np.zeros([1,512])
y=np.zeros([1])
for data in tqdm(train_loader, desc="val..."):
    inputs, outputs = data
    inputs = torch.squeeze(inputs)
    inputs = inputs.to(device)
    _,predicted_outputs = model(inputs)
    predicted_outputs = predicted_outputs.to('cpu')
    predicted = predicted_outputs.detach().numpy()
    x1=np.vstack((x1,predicted[0,:]))
    x2=np.vstack((x2, predicted[1,:]))
    x3=np.vstack((x3, predicted.max(axis=0)))
    y=np.vstack((y,outputs))

reducer = umap.UMAP(n_neighbors=100,min_dist=0.7,metric='correlation',n_components=20,random_state=42)  #5,0.1,20,42
reducer.fit(x1)
embedding = reducer.transform(x1)
assert(np.all(embedding == reducer.embedding_))
embedding.shape
plt.figure(figsize=(8,6))
scatter=plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.rcParams.update({'font.size': 25})
plt.legend(handles=scatter.legend_elements()[0],labels=['Tomor','Normal'],loc=4)
# plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(2))
plt.title('UMAP projection of the Tumor dataset', fontsize=25);
plt.savefig('./umap.pdf')