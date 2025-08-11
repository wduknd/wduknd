import torch
import torch.nn as nn
from torch.nn import functional as F


def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)

# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            criterion = nn.MSELoss()
            loss = criterion(embedded_bg, embedded_fg)

        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)

class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)

def compute_feature_center(features, masks):
    """
    计算目标和背景的特征中心。
    features: [B, C, H, W]
    masks:    [B, 1, H, W]，值为 1 是目标，0 是背景
    """
    B, C, H, W = features.shape

    # 目标区域特征中心
    mask_sum = masks.sum(dim=[2, 3], keepdim=True)  # [B, 1, 1]
    mask_sum_safe = mask_sum.clone()
    mask_sum_safe[mask_sum_safe == 0] = 1.0  # 防止除0
    target_feat_sum = (features * masks).sum(dim=[2, 3], keepdim=True)
    target_features = (target_feat_sum / mask_sum_safe).squeeze(-1).squeeze(-1)  # [B, C]

    # 背景区域特征中心
    bg_masks = 1 - masks
    bg_sum = bg_masks.sum(dim=[2, 3], keepdim=True)
    bg_sum_safe = bg_sum.clone()
    bg_sum_safe[bg_sum_safe == 0] = 1.0
    background_feat_sum = (features * bg_masks).sum(dim=[2, 3], keepdim=True)
    background_features = (background_feat_sum / bg_sum_safe).squeeze(-1).squeeze(-1)  # [B, C]

    return target_features, background_features

def sfa_loss(syn_features, real_features, syn_masks, real_masks):
    """
    计算 SFA Loss：使目标区域和背景区域的合成与真实特征中心对齐
    syn_features / real_features: [B, C, H, W]
    syn_masks / real_masks:       [B, 1, H, W]
    """
    # 计算每个 domain 的特征中心
    syn_target, syn_background = compute_feature_center(syn_features, syn_masks)
    real_target, real_background = compute_feature_center(real_features, real_masks)

    # 按 batch 平均（得到整个 batch 的“域中心”）
    syn_target_center = syn_target.mean(dim=0)
    syn_background_center = syn_background.mean(dim=0)
    real_target_center = real_target.mean(dim=0)
    real_background_center = real_background.mean(dim=0)

    # 欧氏距离（均方差）作为损失
    target_loss = F.mse_loss(syn_target_center, real_target_center)
    background_loss = F.mse_loss(syn_background_center, real_background_center)

    loss = 0.5 * (target_loss + background_loss)

    # 防止 NaN 传播
    if torch.isnan(loss):
        # print("Warning: SFA Loss is NaN!")
        return torch.tensor(0.0, device=loss.device)
    return loss



class SimMaxLoss_1(nn.Module):
    def __init__(self, metric="cos", temp=0.1, alpha=0.25, reduction="mean"):
        super(SimMaxLoss_1, self).__init__()
        self.metric = metric
        self.temp = temp
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_out):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == "l2":
            raise NotImplementedError

        elif self.metric == "cos":
            sim = cos_simi(embedded_out)
            loss = -torch.log(sim)

            loss[loss < 0] = 0

            loss = loss  * self.alpha
        else:
            raise NotImplementedError

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)


# class SimMinLoss(nn.Module):
#     def __init__(self, metric='cos', reduction='mean'):
#         super(SimMinLoss, self).__init__()
#         self.metric = metric
#         self.reduction = reduction

#     def forward(self, embedded_bg, embedded_fg):
#         """
#         :param embedded_fg: [N, C]
#         :param embedded_bg: [N, C]
#         :return:
#         """
#         if self.metric == 'l2':
#             raise NotImplementedError
#         elif self.metric == 'cos':
#             sim = cos_simi(embedded_bg, embedded_fg)
#             loss = -torch.log(1 - sim)
#         else:
#             raise NotImplementedError

#         if self.reduction == 'mean':
#             return torch.mean(loss)
#         elif self.reduction == 'sum':
#             return torch.sum(loss)

# class SimMaxLoss(nn.Module):
#     def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
#         super(SimMaxLoss, self).__init__()
#         self.metric = metric
#         self.alpha = alpha
#         self.reduction = reduction

#     def forward(self, embedded_bg):
#         """
#         :param embedded_fg: [N, C]
#         :param embedded_bg: [N, C]
#         :return:
#         """
#         if self.metric == 'l2':
#             raise NotImplementedError

#         elif self.metric == 'cos':
#             sim = cos_simi(embedded_bg, embedded_bg)
#             loss = -torch.log(sim)
#             loss[loss < 0] = 0
#             _, indices = sim.sort(descending=True, dim=1)
#             _, rank = indices.sort(dim=1)
#             rank = rank - 1
#             rank_weights = torch.exp(-rank.float() * self.alpha)
#             loss = loss * rank_weights
#         else:
#             raise NotImplementedError

#         if self.reduction == 'mean':
#             return torch.mean(loss)
#         elif self.reduction == 'sum':
#             return torch.sum(loss)


# class SimMaxLoss_1(nn.Module):
#     def __init__(self, metric="cos", temp=0.1, alpha=0.25, reduction="mean"):
#         super(SimMaxLoss_1, self).__init__()
#         self.metric = metric
#         self.temp = temp
#         self.alpha = alpha
#         self.reduction = reduction

#     def forward(self, embedded_out):
#         """
#         :param embedded_fg: [N, C]
#         :param embedded_bg: [N, C]
#         :return:
#         """
#         if self.metric == "l2":
#             raise NotImplementedError

#         elif self.metric == "cos":
#             sim = cos_simi(embedded_out)
#             loss = -torch.log(sim)

#             loss[loss < 0] = 0
#             _, indices = sim.sort(descending=True, dim=1)
#             _, rank = indices.sort(dim=1)
#             rank = rank - 1
#             rank_weights = torch.exp(-rank.float() * self.alpha)
#             loss = loss 
#         else:
#             raise NotImplementedError

#         if self.reduction == "mean":
#             return torch.mean(loss)
#         elif self.reduction == "sum":
#             return torch.sum(loss)

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        # print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)

class CVLoss(nn.Module):
    def __init__(self, loss_weight=1.0,reduction='mean'):
        super(CVLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self,logits):
        # print(torch.mean(logits,dim=1).shape)
        cv = torch.std(logits,dim=1)/torch.mean(logits,dim=1)
        # print(cv)
        return self.loss_weight*torch.mean(cv**2)


# class CVLoss(nn.Module):
#     def __init__(self, loss_weight=1.0, reduction='mean'):
#         super(CVLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.reduction = reduction
#         self.eps = 1e-6
    
#     def forward(self, logits):
#         # 将 logits 转换为 float32 确保计算精度
#         logits = logits.float()

#         # 计算变异系数时，确保标准差和均值都是 float32
#         mean_logits = torch.mean(logits, dim=1).float()  # 确保均值为 float32
#         std_logits = torch.std(logits).float()           # 确保标准差为 float32

#         # 计算变异系数 (Coefficient of Variation, CV)
#         cv = std_logits / (mean_logits + self.eps)
        
#         # print(cv)  # 打印 CV 的值，确保输出

#         # 计算损失
#         return self.loss_weight * torch.mean(cv**2)
class CVLoss(nn.Module):
    def __init__(self, loss_weight=1.0,reduction='mean'):
        super(CVLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self,gates):
        # print(torch.mean(logits,dim=1).shape)
        eps = 1e-10
        x = gates.sum(0)
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)