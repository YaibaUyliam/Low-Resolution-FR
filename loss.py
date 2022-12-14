import torch
import torch.nn as nn
import random 
import numpy as np 

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=2, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

def find_pair(feature_ori, feature_lr, len_batch):
    list_ind = [*range(len_batch)] 
    pairs = []
    dis_type = nn.PairwiseDistance()

    while len(list_ind)>2:
        dis_HL = dis_type(feature_lr[list_ind[1:]], feature_ori[list_ind[0]])
        dis_LH = dis_type(feature_ori[list_ind[1:]], feature_lr[list_ind[0]])
        dis_total = dis_HL + dis_LH
        ind_dis_min = torch.argmin(dis_total) + 1
        pairs.append([list_ind[0], list_ind[ind_dis_min]])
        list_ind.pop(ind_dis_min)   
        list_ind.pop(0)
    
    if len(list_ind) == 2:
        pairs.append([list_ind[0], list_ind[1]])
    
    return pairs

def triple_loss(feature_ori, feature_lr):
    loss = 0
    loss_function = nn.TripletMarginLoss(margin=0.1, p=2)
    len_batch = feature_ori.shape[0]
    pairs = find_pair(feature_ori, feature_lr, len_batch)
    feature_ori = torch.unsqueeze(feature_ori, 1)
    feature_lr = torch.unsqueeze(feature_lr, 1)
    for feature_pair in pairs:
        #print(feature_ori[feature_pair[0]].shape)
        loss_h_0 = loss_function(feature_ori[feature_pair[0]], feature_lr[feature_pair[0]], feature_lr[feature_pair[1]])  # shape (, 1)
        loss_l_0 = loss_function(feature_lr[feature_pair[0]], feature_ori[feature_pair[0]], feature_ori[feature_pair[1]])

        loss_h_1 = loss_function(feature_ori[feature_pair[1]], feature_lr[feature_pair[1]], feature_lr[feature_pair[0]])
        loss_l_1 = loss_function(feature_lr[feature_pair[1]], feature_ori[feature_pair[1]], feature_ori[feature_pair[0]])

        loss += loss_h_0 + loss_l_0 + loss_h_1 + loss_l_1

    return loss/len_batch

if __name__ == "__main__":
    img_ori = torch.rand(32, 1024, 1)
    img_lr = torch.rand(32, 1024, 1)
    print(img_lr[0].shape)
    l = triple_loss(img_ori, img_lr)
    print(l)
