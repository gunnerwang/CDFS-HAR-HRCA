import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils import split_first_dim_linear
from losses import prototype_loss, cross_entropy_loss
from SAG.dual_resnet import resnet18 as dual_resnet18
import math
import copy
from itertools import combinations 

from torch.autograd import Variable

import torchvision.models as models

NUM_SAMPLES=1

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        
        # static pe
        support_set = self.pe(support_set)   # (way*shot)*seqlen*512
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)    # (way*shot)*comb*1024
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way).cuda()

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))

            class_k = class_k.transpose(-2,-1)
            if hasattr(self, 'ia3'):
                class_k, class_v = self.ia3(class_k, class_v)
            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k) / math.sqrt(self.args.trans_linear_out_dim)
            
            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0,2,1,3)
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2,-1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict



    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask) 
        return torch.reshape(class_mask_indices, (-1,))
        

class conv_tsa(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa, self).__init__()
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.alpha.requires_grad = True

    def forward(self, x):
        y = self.conv(x)
        if self.alpha.size(0) > 1:
            y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
        else:
            y = y + x * self.alpha
        return y


class pa(nn.Module):
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        return x


class trans_ia3(nn.Module):
    def __init__(self, feat_dim):
        super(trans_ia3, self).__init__()
        self.weight_k = nn.Parameter(torch.ones(1, feat_dim, 1))
        self.weight_v = nn.Parameter(torch.ones(1, 1, feat_dim))
        self.weight_k.requires_grad = True
        self.weight_v.requires_grad = True

    def forward(self, clk, clv):
        clk = self.weight_k.to(clk.device) * clk
        clv = self.weight_v.to(clv.device) * clv
        return clk, clv


class CNN_TRX(nn.Module):
    def __init__(self, args):
        super(CNN_TRX, self).__init__()

        self.train()
        self.args = args

        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        if self.args.modality == ['rgb']:
            last_layer_idx = -1
            self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        elif self.args.modality == ['rgb', 'depth']:
            # last_layer_idx = -1
            self.resnet = dual_resnet18(pretrained_model='./SAG/resnet18-f37072fd.pth')
            # self.resnet = nn.Sequential(*list(resnet.children()))

        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 

    def forward(self, context_images, context_labels, target_images):
        if self.args.modality == ['rgb']:
            context_features = self.resnet(context_images).squeeze()
            target_features = self.resnet(target_images).squeeze()
        elif self.args.modality == ['rgb', 'depth']:
            # context_features_rgb = self.resnet(context_images[:, :3, :, :]).squeeze()
            # target_features_rgb = self.resnet(target_images[:, :3, :, :]).squeeze()
            # context_features_depth = self.resnet(torch.repeat_interleave(context_images[:, 3:, :, :], 3, dim=1)).squeeze()
            # target_features_depth = self.resnet(torch.repeat_interleave(target_images[:, 3:, :, :], 3, dim=1)).squeeze()
            context_features = self.resnet(context_images[:, :3, :, :], torch.repeat_interleave(context_images[:, 3:, :, :], 3, dim=1)).squeeze()
            target_features = self.resnet(target_images[:, :3, :, :], torch.repeat_interleave(target_images[:, 3:, :, :], 3, dim=1)).squeeze()

        if hasattr(self, 'beta'):
            context_features, target_features = self.beta(context_features), self.beta(target_features)

        dim = int(context_features.shape[1])

        context_features = context_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        all_logits = [t(context_features, context_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_features.shape[0]])}
        return return_dict

    def distribute_model(self):
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)
    
    def add_adaptor(self):
        for k, v in self.named_parameters():
            v.requires_grad=False

        for block in self.resnet[4]:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in self.resnet[5]:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in self.resnet[6]:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for block in self.resnet[7]:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m)
                    setattr(block, name, new_conv)

        for t in self.transformers:
            ia3 = trans_ia3(self.args.trans_linear_out_dim) 
            setattr(t, 'ia3', ia3)

        feat_dim = self.resnet[7][-1].bn2.num_features
        beta = pa(feat_dim)
        setattr(self, 'beta', beta)

    def reset_adaptor(self):
        for k, v in self.resnet.named_parameters():
            if 'alpha' in k:
                if v.size(0) > 1:
                    v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)
                else:
                    v.data = torch.ones(v.size()).to(v.device)
                v.data = v.data * 0.0001
                if 'bias' in k:
                    v.data = v.data * 0
        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)


def tsa(context_images, context_labels, model, max_iter=40, lr_alpha=1e-4, lr_beta=1e-4, lr=1e-4):
    model.eval()
    alpha_params = [v for k, v in model.named_parameters() if 'alpha' in k]
    beta_params = [v for k, v in model.named_parameters() if 'beta' in k]
    ia3_params = [v for k, v in model.named_parameters() if 'ia3' in k]
    params = []
    params.append({'params': alpha_params, 'lr': lr_alpha})
    params.append({'params': beta_params, 'lr': lr_beta})
    params.append({'params': ia3_params})

    optimizer = torch.optim.Adam(params, lr=lr) 

    for i in range(max_iter):
        optimizer.zero_grad()
        model.zero_grad()

        context_logits = model(context_images, context_labels, context_images)['logits'].squeeze(0)
        loss, _, _ = cross_entropy_loss(context_logits, context_labels)

        loss.backward()
        optimizer.step()
    return


if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 1
            self.query_per_class = 5
            self.trans_dropout = 0.1
            self.seq_len = 8 
            self.img_size = 84
            self.method = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2,3]

    args = ArgsObject()
    torch.manual_seed(0)
    
    device = 'cuda:0'
    model = CNN_TRX(args)
    model.add_adaptor()
    model.reset_adaptor()
    model = model.to(device)
    
    support_imgs = torch.rand(args.way * args.shot * args.seq_len,3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len ,3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0,1,2,3,4]).to(device)

    print("Support images input shape: {}".format(support_imgs.shape))
    print("Target images input shape: {}".format(target_imgs.shape))
    print("Support labels input shape: {}".format(support_labels.shape))

    tsa(support_imgs, support_labels, model, max_iter=40, lr=0.5, lr_beta=1.0)

    out = model(support_imgs, support_labels, target_imgs)

    print("Logits shape: {}".format(out['logits'].shape))





