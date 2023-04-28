import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import torchvision.models as models
from utils.utils import initialize_weights


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((0, 1), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((0, 1), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)
        gate=torch.squeeze(gate)
        return x * gate

class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes

# My Net
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=256, D=512, dropout=False, n_classes=21, att=False):
        super(Attn_Net_Gated, self).__init__()
        self.att = att
        self.attention_a = [nn.Linear(L, D),
                            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)  # W

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)  # 点乘
        A = self.attention_c(A)  # N x n_classes => num_patch × n_classes
        if self.att:
            A = A.mean(dim=1)
        return A, x


class CLAM_SB(nn.Module):
    def __init__(self, size_arg="small", dropout=False, n_classes=21):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]  # size = [1024,512,256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        classifiers = [nn.Linear(size[1], 256), nn.ReLU(), nn.Linear(256, n_classes)]
        self.classifiers = nn.Sequential(*classifiers)

        self.n_classes = n_classes

        initialize_weights(self)

    def relocate(self, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h, attention_only=False):

        A, h = self.attention_net(h)  # NxK     A:batch_size × num_patches × 512 , h: 原始输入
        A = torch.transpose(A, 2, 1)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=2)  # softmax over N
        # c = torch.sum(A, dim=2)
        A = A.squeeze()
        h = h.squeeze()
        M = torch.mm(A, h)  # 乘权重
        M = M.mean(dim=0)
        # print(M.shape)
        logits = self.classifiers(M)

        return logits, A_raw

class CLAM_SB_Class(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=21,
                 freeze=False):
        super(CLAM_SB_Class, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [2048, 512, 256]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)  # size[1]=512, n_classes=10
        self.n_classes = n_classes

        initialize_weights(self)
        if freeze:
            self.attention_net.requires_grad_(False)

    def relocate(self, device):
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
       # self.instance_classifiers = self.instance_classifiers.to(device)

    def forward(self, h,attention_only=False):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)
        logits = self.classifiers(M)  # 返回分类层的输出 维度=n_classes
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Y_prob = F.softmax(logits, dim=1)
        # if instance_eval:
        #     results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
        #                     'inst_preds': np.array(all_preds)}
        # else:
        #     results_dict = {}
        return logits,M
class CLAM_SB_TIME_NN_Pool(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=1,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, freeze=False, N=100,
                 size_dict = {"small": [1024, 512, 256], "big": [2048, 512, 256]}
                 ):
        super(CLAM_SB_TIME_NN_Pool, self).__init__()
        self.size_dict = size_dict
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1] * 2, n_classes)
        self.n_classes = n_classes
        self.N = N

        initialize_weights(self)
        if freeze:
            self.attention_net.requires_grad_(False)

    def relocate(self, device):
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)


    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N

        if h.shape[0] > self.N * 2:
            idxs = torch.argsort(A[0])
            low_n_idxs = idxs[:self.N]
            high_n_idxs = idxs[-self.N:]

            low_n = h[low_n_idxs].mean(axis=0)
            high_n = h[high_n_idxs].mean(axis=0)

            M = torch.cat([low_n, high_n])
            M = torch.unsqueeze(M, 0)
        else:
            M = torch.mm(A, h)
            M = torch.concat([M[0], M[0]])
            M = torch.unsqueeze(M, 0)

        if return_features:
            return M
        logits = self.classifiers(M)
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Y_prob = F.softmax(logits, dim=1)
        # if instance_eval:
        #     results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
        #                     'inst_preds': np.array(all_preds)}
        # else:
        #     results_dict = {}
        del A, M, h
        gc.collect()

        return logits

class CLAM_MB_TIME(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, mlp=0, n_classes=21,
                 ):
        super(CLAM_MB_TIME, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [2048, 512, 256]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention)
        self.attention_net = nn.Sequential(*fc)
        if mlp == 0:
            bag_classifiers = [nn.Linear(size[1], 1) for i in
                               range(n_classes)] # use an indepdent linear layer to predict each class
        else:
            bag_classifiers = [nn.Sequential(nn.Linear(size[1], mlp), nn.ReLU(),nn.Dropout(0.25), nn.Linear(mlp, 1))
                               for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)

        self.n_classes = n_classes
        self.n_tasks = n_classes
        initialize_weights(self)

    def forward(self, h,  attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        del A, M, h
        gc.collect()
        return logits

    def get_last_shared_layer(self):
        return self.attention_net[3].attention_c

class CLAM_MB_TIME_Gradnorm(torch.nn.Module):
    
    def __init__(self, model):
        
        
        super(CLAM_MB_TIME_Gradnorm, self).__init__()
        self.model = model
        self.weights = torch.nn.Parameter(torch.ones(model.n_tasks).float())
        self.mse_loss = MSELoss()
        
    def forward(self, x, ts,attention_only=False):
        n_tasks = self.model.n_tasks
        ys = self.model(x)

        task_loss = []
        if not attention_only:
            # check if the number of tasks is equal to this size
            assert (ys.size()[1] == n_tasks)
            task_loss = []
            for i in range(n_tasks):
                task_loss.append(self.mse_loss(ys[:, i], ts[:, i]))
            task_loss = torch.stack(task_loss)
    
        return ys,task_loss

    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()

class CLAM_MB_Classify(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, mlp=0, n_classes=21,
                 ):
        super(CLAM_MB_Classify, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [2048, 512, 256]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        if mlp == 0:
            bag_classifiers = [nn.Linear(size[1], 1) for i in
                               range(n_classes)]  # use an indepdent linear layer to predict each class
        else:
            bag_classifiers = [nn.Sequential(nn.Linear(size[1], mlp), nn.ReLU(), nn.Linear(mlp, 1))
                               for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        
        self.n_classes = n_classes
        initialize_weights(self)
    
    def forward(self, h, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        if return_features:
            results_dict = {}
            results_dict.update({'features': M})
        del A, M, h
        gc.collect()
        return logits


class CLAM_MB_Reg_gct(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, mlp=0, n_classes=21,
                 ):
        super(CLAM_MB_Reg_gct, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [2048, 512, 256]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), GCT(size[1]),nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention)
        self.attention_net = nn.Sequential(*fc)
        if mlp == 0:
            bag_classifiers = [nn.Linear(size[1], 1) for i in
                               range(n_classes)]  # use an indepdent linear layer to predict each class
        else:
            bag_classifiers = [nn.Sequential(nn.Linear(size[1], mlp), nn.ReLU(), nn.Dropout(0.25), nn.Linear(mlp, 1))
                               for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)

        self.n_classes = n_classes
        self.n_tasks = n_classes
        initialize_weights(self)

    def forward(self, h, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        del A, M, h
        gc.collect()
        return logits

    def get_last_shared_layer(self):
        return self.attention_net[3].attention_c

if __name__ == '__main__':
    model = CLAM_MB_TIME()
    test_inp = torch.randn(1000 * 1024).reshape(1000, 1024)
    test_label = torch.randn(21)
    logits = model(test_inp)
    nn.MSELoss(test_inp,test_label,reduction='sum')