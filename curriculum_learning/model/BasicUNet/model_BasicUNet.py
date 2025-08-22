import torch
import torch.nn as nn
from itertools import repeat
from torch import Tensor
from torchvision import models, ops
import numpy as np
import torch.nn.functional as F
from .dysample import DySample
from .moe  import Decoder as moe 
k11x11 = [[2.79977845179169e-05, 0.000106801114075605, 0.000302563348459321, 0.000636568686406847, 0.000994632494399497, 0.00115416641889010, 0.000994632494399497, 0.000636568686406847, 0.000302563348459321, 0.000106801114075605, 2.79977845179169e-05], 
          [0.000106801114075605, 0.000407406448909944, 0.00115416641889010, 0.00242827231027471, 0.00379415229907511, 0.00440271476792141, 0.00379415229907511, 0.00242827231027471, 0.00115416641889010, 0.000407406448909944, 0.000106801114075605], 
          [0.000302563348459321, 0.00115416641889010, 0.00326970799322831, 0.00687919978669574, 0.0107486839824550, 0.0124727174807428, 0.0107486839824550, 0.00687919978669574, 0.00326970799322831, 0.00115416641889010, 0.000302563348459321], 
          [0.000636568686406847, 0.00242827231027471, 0.00687919978669574, 0.0144732770642771, 0.0226143572185962, 0.0262415835330685, 0.0226143572185962, 0.0144732770642771, 0.00687919978669574, 0.00242827231027471, 0.000636568686406847], 
          [0.000994632494399497, 0.00379415229907511, 0.0107486839824550, 0.0226143572185962, 0.0353347172267247, 0.0410022237094569, 0.0353347172267247, 0.0226143572185962, 0.0107486839824550, 0.00379415229907511, 0.000994632494399497], 
          [0.00115416641889010, 0.00440271476792141, 0.0124727174807428, 0.0262415835330685, 0.0410022237094569, 0.0475787690144247, 0.0410022237094569, 0.0262415835330685, 0.0124727174807428, 0.00440271476792141, 0.00115416641889010], 
          [0.000994632494399497, 0.00379415229907511, 0.0107486839824550, 0.0226143572185962, 0.0353347172267247, 0.0410022237094569, 0.0353347172267247, 0.0226143572185962, 0.0107486839824550, 0.00379415229907511, 0.000994632494399497], 
          [0.000636568686406847, 0.00242827231027471, 0.00687919978669574, 0.0144732770642771, 0.0226143572185962, 0.0262415835330685, 0.0226143572185962, 0.0144732770642771, 0.00687919978669574, 0.00242827231027471, 0.000636568686406847], 
          [0.000302563348459321, 0.00115416641889010, 0.00326970799322831, 0.00687919978669574, 0.0107486839824550, 0.0124727174807428, 0.0107486839824550, 0.00687919978669574, 0.00326970799322831, 0.00115416641889010, 0.000302563348459321], 
          [0.000106801114075605, 0.000407406448909944, 0.00115416641889010, 0.00242827231027471, 0.00379415229907511, 0.00440271476792141, 0.00379415229907511, 0.00242827231027471, 0.00115416641889010, 0.000407406448909944, 0.000106801114075605], 
          [2.79977845179169e-05, 0.000106801114075605, 0.000302563348459321, 0.000636568686406847, 0.000994632494399497, 0.00115416641889010, 0.000994632494399497, 0.000636568686406847, 0.000302563348459321, 0.000106801114075605, 2.79977845179169e-05]]

k9x9 = [[5.79793792857477e-05, 0.000274689940733107, 0.000834434361929352, 0.00162525621175554, 0.00202969938188886, 0.00162525621175554, 0.000834434361929352, 0.000274689940733107, 5.79793792857477e-05], 
        [0.000274689940733107, 0.00130140343807554, 0.00395331457921257, 0.00770000538093006, 0.00961614301128137, 0.00770000538093006, 0.00395331457921257, 0.00130140343807554, 0.000274689940733107], 
        [0.000834434361929352, 0.00395331457921257, 0.0120091093237972, 0.0233905510327068, 0.0292112632025221, 0.0233905510327068, 0.0120091093237972, 0.00395331457921257, 0.000834434361929352], 
        [0.00162525621175554, 0.00770000538093006, 0.0233905510327068, 0.0455585724854297, 0.0568957717217601, 0.0455585724854297, 0.0233905510327068, 0.00770000538093006, 0.00162525621175554], 
        [0.00202969938188886, 0.00961614301128137, 0.0292112632025221, 0.0568957717217601, 0.0710542201656980, 0.0568957717217601, 0.0292112632025221, 0.00961614301128137, 0.00202969938188886], 
        [0.00162525621175554, 0.00770000538093006, 0.0233905510327068, 0.0455585724854297, 0.0568957717217601, 0.0455585724854297, 0.0233905510327068, 0.00770000538093006, 0.00162525621175554], 
        [0.000834434361929352, 0.00395331457921257, 0.0120091093237972, 0.0233905510327068, 0.0292112632025221, 0.0233905510327068, 0.0120091093237972, 0.00395331457921257, 0.000834434361929352], 
        [0.000274689940733107, 0.00130140343807554, 0.00395331457921257, 0.00770000538093006, 0.00961614301128137, 0.00770000538093006, 0.00395331457921257, 0.00130140343807554, 0.000274689940733107], 
        [5.79793792857477e-05, 0.000274689940733107, 0.000834434361929352, 0.00162525621175554, 0.00202969938188886, 0.00162525621175554, 0.000834434361929352, 0.000274689940733107, 5.79793792857477e-05]]

k7x7 = [[0.000157758635808270, 0.000990095038924366, 0.00298048629897427, 0.00430352052106079, 0.00298048629897427, 0.000990095038924366, 0.000157758635808270],
        [0.000990095038924366, 0.00621384801586408, 0.0187055667861054, 0.0270089449999423, 0.0187055667861054, 0.00621384801586408, 0.000990095038924366],
        [0.00298048629897427, 0.0187055667861054, 0.0563094282151982, 0.0813051145166149, 0.0563094282151982, 0.0187055667861054, 0.00298048629897427], 
        [0.00430352052106079, 0.0270089449999423, 0.0813051145166149, 0.117396355390014, 0.0813051145166149, 0.0270089449999423, 0.00430352052106079], 
        [0.00298048629897427, 0.0187055667861054, 0.0563094282151982, 0.0813051145166149, 0.0563094282151982, 0.0187055667861054, 0.00298048629897427], 
        [0.000990095038924366, 0.00621384801586408, 0.0187055667861054, 0.0270089449999423, 0.0187055667861054, 0.00621384801586408, 0.000990095038924366], 
        [0.000157758635808270, 0.000990095038924366, 0.00298048629897427, 0.00430352052106079, 0.00298048629897427, 0.000990095038924366, 0.000157758635808270]]

k5x5 = [[0.000724317986003452, 0.00628066096513693, 0.0129031984715720, 0.00628066096513693, 0.000724317986003452], 
        [0.00628066096513693, 0.0544604758148402, 0.111885410181476, 0.0544604758148402, 0.00628066096513693],
        [0.0129031984715720, 0.111885410181476, 0.229861102463338, 0.111885410181476, 0.0129031984715720], 
        [0.00628066096513693, 0.0544604758148402, 0.111885410181476, 0.0544604758148402, 0.00628066096513693], 
        [0.000724317986003452, 0.00628066096513693, 0.0129031984715720, 0.00628066096513693, 0.000724317986003452]]

k3x3 = [[0.0113437365584951, 0.0838195058022106, 0.0113437365584951], 
        [0.0838195058022106, 0.619347030557177, 0.0838195058022106], 
        [0.0113437365584951, 0.0838195058022106, 0.0113437365584951]]

class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        if self.training:   # for better optimizatioin
            avg_out = self.fc2(self.relu1(self.fc1(nn.functional.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3])))))
            max_out = self.fc2(self.relu1(self.fc1(nn.functional.max_pool2d(x, kernel_size=(x.shape[2], x.shape[3])))))
        else:
            avg_out = self.fc2(self.relu1(self.fc1(x.flatten(2).mean(dim=2, keepdim=True).unsqueeze(3))))  # for fast inference # 这个和上面的结果不一致，不要轻易改变！
            max_out = self.fc2(self.relu1(self.fc1(x.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(3))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# class DropBlock(nn.Module):
#     def __init__(self, block_size: int, p: float = 0.5):
#         super().__init__()
#         self.block_size = block_size
#         self.p = p


#     def calculate_gamma(self, x: Tensor) -> float:
#         """Compute gamma, eq (1) in the paper
#         Args:
#             x (Tensor): Input tensor
#         Returns:
#             Tensor: gamma
#         """
        
#         invalid = (1 - self.p) / (self.block_size ** 2)
#         valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
#         return invalid * valid

#     def forward(self, x: Tensor) -> Tensor:
#         if self.training:
#             gamma = self.calculate_gamma(x)
#             mask = torch.bernoulli(torch.ones_like(x) * gamma)
#             mask_block = 1 - F.max_pool2d(
#                 mask,
#                 kernel_size=(self.block_size, self.block_size),
#                 stride=(1, 1),
#                 padding=(self.block_size // 2, self.block_size // 2),
#             )
#             x = mask_block * x * (mask_block.numel() / mask_block.sum())
#         return x




class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """
    def __init__(self, drop=0.1):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], 1, inputs.shape[2],inputs.shape[-1])   # 默认沿着Channel的shape
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)



class FrequencyAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(FrequencyAttention, self).__init__()

        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        xfft = torch.fft.fft2(x)
        xfft_real, xfft_imag = xfft.real, xfft.imag

        out_real = self.ca(xfft_real) * xfft_real
        out_imag = self.ca(xfft_imag) * xfft_imag
        out_fft = torch.complex(out_real, out_imag)
        out = torch.fft.ifft2(out_fft)
        
        return out.real
    

class Res_CBAM_block_mscn(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block_mscn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        window_kernel = np.asarray(k7x7)  # defaut 7x7
        assert window_kernel.shape[0] == window_kernel.shape[1]
        ks = window_kernel.shape[0]

        window_kernel = torch.FloatTensor(window_kernel).expand(out_channels, 1, ks, ks)
        self.padding = int(ks // 2)
        # print('k: ', ks)

        self.window = nn.Parameter(data=window_kernel, requires_grad=False)  # default False

        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)  # default 0.5
        # self.alpha = 0.5
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.moe = moe(out_channels,4,2)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        # out += residual
        mu = torch.nn.functional.conv2d(residual, self.window, padding=self.padding, groups=residual.shape[1])
        mu_sq = mu * mu
        sigma = torch.sqrt(torch.abs(torch.nn.functional.conv2d(residual * residual, self.window, padding=self.padding, groups=residual.shape[1]) - mu_sq) + 1e-8)
        mscn = (residual - mu) / (sigma + 1)
        mscn = self.bn3(mscn)

        out += (residual + self.alpha * mscn)
        # -----------------------------------

        out = self.relu(out)
        return out

class Res_CBAM_block_mscn_moe(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block_mscn_moe, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        window_kernel = np.asarray(k7x7)  # defaut 7x7
        assert window_kernel.shape[0] == window_kernel.shape[1]
        ks = window_kernel.shape[0]

        window_kernel = torch.FloatTensor(window_kernel).expand(out_channels, 1, ks, ks)
        self.padding = int(ks // 2)
        # print('k: ', ks)

        self.window = nn.Parameter(data=window_kernel, requires_grad=False)  # default False

        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)  # default 0.5
        # self.alpha = 0.5
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        # out += residual
        mu = torch.nn.functional.conv2d(residual, self.window, padding=self.padding, groups=residual.shape[1])
        mu_sq = mu * mu
        sigma = torch.sqrt(torch.abs(torch.nn.functional.conv2d(residual * residual, self.window, padding=self.padding, groups=residual.shape[1]) - mu_sq) + 1e-8)
        mscn = (residual - mu) / (sigma + 1)
        mscn = self.bn3(mscn)

        out += (residual + self.alpha * mscn)
        # -----------------------------------

        out = self.relu(out)
        return out


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out


class Res_SAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_SAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        # self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.ca(out) * out
        out = self.sa(out) * out
        
        out += residual
        out = self.relu(out)
        return out
    

class Res_CAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        # self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        # out = self.sa(out) * out
        
        out += residual
        out = self.relu(out)
        return out


class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out
        
        out += residual

        # out_ca = self.ca(out) * out
        # out_sa = self.ca(out) * out
        # out = residual + out_ca + out_sa

        out = self.relu(out)
        return out

class Res_CBAM_dropblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_dropblock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.drop1 = ops.DropBlock2d(p=0.1, block_size=7)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.drop2 = ops.DropBlock2d(p=0.1, block_size=7)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.drop2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        out += residual

        # out_ca = self.ca(out) * out
        # out_sa = self.ca(out) * out
        # out = residual + out_ca + out_sa

        out = self.relu(out)
        return out
    
class Res_CBAM_spatialdrop(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_spatialdrop, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.drop1 = SpatialDropout(drop=0.1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

        self.drop2 = SpatialDropout(drop=0.1)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        out += residual

        # out_ca = self.ca(out) * out
        # out_sa = self.ca(out) * out
        # out = residual + out_ca + out_sa

        out = self.relu(out)
        out = self.drop2(out)
        return out

class Res_CBAM_fft_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_fft_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ffta = FrequencyAttention(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        out = self.ffta(out)

        out += residual

        # out_ca = self.ca(out) * out
        # out_sa = self.ca(out) * out
        # out = residual + out_ca + out_sa

        out = self.relu(out)
        return out

class BasicUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256], deep_supervision=True):
        super(BasicUNet, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        xf = self.conv0_1(torch.cat([xu3, xu2, xu1, xu0, x0_1], 1))

        if self.deep_supervision:
            output1 = self.final1(xu3).sigmoid()
            output2 = self.final2(xu2).sigmoid()
            output3 = self.final3(xu1).sigmoid()
            output4 = self.final4(xu0).sigmoid()
            output5 = self.final5(xf).sigmoid()
            if self.training:
                return [output1, output2, output3, output4, output5]
            else:
                return output5
        else:
            output = self.final(xf).sigmoid()
            return output



class BasicUNet_plus(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        return output
        # if self.training:
        #     return output, x4_0
        # else:
        #     return output


class BasicUNet_plus_Feature(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_Feature, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input,return_features=False):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)
        output = self.final(xf).sigmoid()
        if return_features:
            feats = [x0_0,x1_0,x2_0,x3_0,x4_0]
            return output, feats
        else:
            return output
    
class BasicUNet_plus_woCBAM(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_woCBAM, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        output = self.final(xf).sigmoid()
        return output

class BasicUNet_plus_woHFFM(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_woHFFM, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        # self.up4   = nn.Upsample(scale_factor=4)
        # self.up8   = nn.Upsample(scale_factor=8)
        # self.up16  = nn.Upsample(scale_factor=16)
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        # self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        # self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        # self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        # self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        # self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        output = self.final(x0_1).sigmoid()
        return output
    
class BasicUNet_plus_woBOTH(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_woBOTH, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        # self.up4   = nn.Upsample(scale_factor=4)
        # self.up8   = nn.Upsample(scale_factor=8)
        # self.up16  = nn.Upsample(scale_factor=16)
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        # self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        # self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        # self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        # self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        # self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        output = self.final(x0_1).sigmoid()
        return output

class BasicUNet_plus_dropblock(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_dropblock, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_dropblock, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        if self.training:
            return output, x4_0
        else:
            return output


class BasicUNet_plus_Spatialdrop(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_spatialdrop, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_Spatialdrop, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        if self.training:
            return output, x4_0
        else:
            return output
        

class BasicUNet_fft(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_fft_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_fft, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        
        return output


class BasicUNet_plus2(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus2, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv4_1_1x1 = nn.Conv2d(nb_filter[3], nb_filter[4], kernel_size=1, stride=1)
        self.conv3_1 = self._make_layer(block, nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv3_1_1x1 = nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=1, stride=1)
        self.conv2_1 = self._make_layer(block, nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv2_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[2], kernel_size=1, stride=1)
        self.conv1_1 = self._make_layer(block, nb_filter[1], nb_filter[0])
        self.conv1_1_1x1 = nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=1, stride=1)

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(self.conv4_1_1x1(x3_0) + self.up(x4_0))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(self.conv3_1_1x1(x2_0) + self.up(x3_1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(self.conv2_1_1x1(x1_0) + self.up(x2_1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(self.conv1_1_1x1(x0_0) + self.up(x1_1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        return output



class BasicUNet_pureRes(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CAM_block, # Res_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_pureRes, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        return output



class BasicUNet_Simple(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256], deep_supervision=True):
        super(BasicUNet_Simple, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        # self.up4   = nn.Upsample(scale_factor=4)
        # self.up8   = nn.Upsample(scale_factor=8)
        # self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        # self.conv0_1 = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        # self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        # self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        # self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        # self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        # xu3 = self.conv0_4_1x1(self.up16(x4_0))
        # xu2 = self.conv0_3_1x1(self.up8(x3_1))
        # xu1 = self.conv0_2_1x1(self.up4(x2_1))
        # xu0 = self.conv0_1_1x1(self.up(x1_1))

        # xf = self.conv0_1(torch.cat([xu3, xu2, xu1, xu0, x0_1], 1))

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
            # output = self.final(xf).sigmoid()
            # return output

        output = self.final(x0_1).sigmoid()
        return output
    



class BasicUNet_plus_dysample(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_dysample, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        # self.up    = nn.Upsample(scale_factor=2)
        # self.up4   = nn.Upsample(scale_factor=4)
        # self.up8   = nn.Upsample(scale_factor=8)
        # self.up16  = nn.Upsample(scale_factor=16)


        self.up_4_0   = DySample(in_channels=nb_filter[4] ,scale=2)
        self.up_3_1   = DySample(in_channels=nb_filter[3] ,scale=2)
        self.up_2_1   = DySample(in_channels=nb_filter[2] ,scale=2)
        self.up_1_1   = DySample(in_channels=nb_filter[1] ,scale=2)
        

        self.up2_1_1   = DySample(in_channels=nb_filter[1] ,scale=2)
        self.up4_2_1   = DySample(in_channels=nb_filter[2] ,scale=4)
        self.up8_3_1   = DySample(in_channels=nb_filter[3] ,scale=8)
        self.up16_4_0  = DySample(in_channels=nb_filter[4] ,scale=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16_4_0(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up_4_0(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8_3_1(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up_3_1(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4_2_1(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up_2_1(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up2_1_1(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up_1_1(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        return output
        # if self.training:
        #     return output, x4_0
        # else:
        #     return output


class ContraModuleNew(nn.Module):
    def __init__(self, channel, dim=128):
        super(ContraModuleNew, self).__init__()

        self.activation_head = nn.Conv2d(channel, channel, 1, padding=0,bias=False)
        self.bn_head = nn.BatchNorm2d(256)
        self.mlp = nn.Sequential(nn.Linear(16*16, 64), nn.ReLU())
        # self.b_mlp = nn.Sequential(nn.Linear(128*128, dim), nn.ReLU())

    def forward(self, x):
        # x: feature maps (output of U-Net)
        ccam = self.bn_head(self.activation_head(x))
        N, C, H, W = ccam.size()
        
        ccam_ = ccam.reshape(N, C, H * W)
        # x = x.reshape()
        # ccam_ = self.mlp(ccam_)
        return ccam_


class ContraModuleNew_2(nn.Module):
    def __init__(self, channel, dim=128):
        super(ContraModuleNew_2, self).__init__()

        self.activation_head = nn.Conv2d(channel, channel, 1, padding=0,bias=False)
        self.bn_head = nn.BatchNorm2d(256)
        # self.mlp = nn.Sequential(nn.Linear(16*16, 64), nn.ReLU())
        # self.b_mlp = nn.Sequential(nn.Linear(128*128, dim), nn.ReLU())

    def forward(self, x):
        # x: feature maps (output of U-Net)
        ccam = self.bn_head(self.activation_head(x))
        N, C, H, W = ccam.size()
        
        ccam_ = ccam.reshape(N, -1)
        # x = x.reshape()
        # ccam_ = self.mlp(ccam_)
        return ccam_


class ContraModuleNew3(nn.Module):
    def __init__(self, channel, dim=128):
        super(ContraModuleNew3, self).__init__()

        self.activation_head = nn.Conv2d(channel, channel, 1, padding=0,bias=False)
        self.bn_head = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.activation_head2 = nn.Conv2d(channel, channel, 1, padding=0,bias=False)

        # self.mlp = nn.Sequential(nn.Linear(16*16, 64), nn.ReLU())
        # self.b_mlp = nn.Sequential(nn.Linear(128*128, dim), nn.ReLU())

    def forward(self, x):
        # x: feature maps (output of U-Net)
        ccam =  self.activation_head2(self.relu(self.bn_head(self.activation_head(x))))

        N, C, H, W = ccam.size()
        
        ccam_ = ccam.reshape(N, C, H * W)
        ccam = ccam.transpose(1, 2)
        # x = x.reshape()
        # ccam_ = self.mlp(ccam_)
        return ccam_

class ContraModuleNew4(nn.Module):
    def __init__(self, channel, dim=128):
        super(ContraModuleNew4, self).__init__()

        self.activation_head = nn.Conv2d(channel, channel, 1, padding=0,bias=False)
        self.bn_head = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.activation_head2 = nn.Conv2d(channel, channel, 1, padding=0,bias=False)

        # self.mlp = nn.Sequential(nn.Linear(16*16, 64), nn.ReLU())
        # self.b_mlp = nn.Sequential(nn.Linear(128*128, dim), nn.ReLU())

    def forward(self, x):
        # x: feature maps (output of U-Net)
        ccam =  self.activation_head2(self.relu(self.bn_head(self.activation_head(x))))

        N, C, H, W = ccam.size()
        # 全局最大池化操作，将每个通道的 H, W 维度池化为 1x1
        global_max_pool = F.adaptive_max_pool2d(ccam, (1, 1))

        # 全局平均池化操作，将每个通道的 H, W 维度池化为 1x1
        global_avg_pool = F.adaptive_avg_pool2d(ccam, (1, 1))  

        global_max_pool = global_max_pool.contiguous().view(N, C)
        global_avg_pool = global_avg_pool.contiguous().view(N, C)
        output = torch.stack([global_max_pool, global_avg_pool], dim=1)

        # ccam_ = ccam.reshape(N, C, H * W)
        # ccam = ccam.transpose(1, 2)
        # x = x.reshape()
        # ccam_ = self.mlp(ccam_)
        return output

class ContraModuleNew5(nn.Module):
    def __init__(self, channel, dim=128):
        super(ContraModuleNew5, self).__init__()

        self.activation_head = nn.Conv2d(channel, channel, 1, padding=0,bias=False)
        self.bn_head = nn.BatchNorm2d(256)
        self.mlp = nn.Sequential(nn.Linear(16*16, 64), nn.ReLU())
        # self.b_mlp = nn.Sequential(nn.Linear(128*128, dim), nn.ReLU())

    def forward(self, x):
        # x: feature maps (output of U-Net)
        ccam = self.bn_head(self.activation_head(x))
        N, C, H, W = ccam.size()
        
        ccam_ = ccam.reshape(N, C, H * W)
        ccam = ccam.transpose(1, 2)
        # x = x.reshape()
        # ccam_ = self.mlp(ccam_)
        return ccam_   

class ContraModuleNew6(nn.Module):
    def __init__(self, channel, dim=128):
        super(ContraModuleNew6, self).__init__()

        self.activation_head = nn.Conv2d(channel, channel, 1, padding=0,bias=False)
        self.bn_head = nn.BatchNorm2d(256)
        self.mlp = nn.Sequential(nn.Linear(16*16, 64), nn.ReLU())
        # self.b_mlp = nn.Sequential(nn.Linear(128*128, dim), nn.ReLU())

    def forward(self, x):
        # x: feature maps (output of U-Net)
        ccam = self.bn_head(self.activation_head(x))
        N, C, H, W = ccam.size()
        # 全局最大池化操作，将每个通道的 H, W 维度池化为 1x1
        global_max_pool = F.adaptive_max_pool2d(ccam, (1, 1))

        # 全局平均池化操作，将每个通道的 H, W 维度池化为 1x1
        global_avg_pool = F.adaptive_avg_pool2d(ccam, (1, 1))  

        global_max_pool = global_max_pool.contiguous().view(N, C)
        global_avg_pool = global_avg_pool.contiguous().view(N, C)
        output = torch.stack([global_max_pool, global_avg_pool], dim=1)

        # ccam_ = ccam.reshape(N, C, H * W)
        # ccam = ccam.transpose(1, 2)
        # x = x.reshape()
        # ccam_ = self.mlp(ccam_)
        return output
    

class BasicUNet_plus_cons(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_cons, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        if self.training:
            return output, self.up16(x4_0)
        else:
            return output

class BasicUNet_plus_moe(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_moe, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        self.decoder = moe(nb_filter[0],4,2)
    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        x_moe, dec_gate = self.decoder(torch.cat([xu3, xu2, xu1, xu0, x0_1], 1))
        # xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(x_moe).sigmoid()
        if self.training:
            return output, x4_0,dec_gate
        else:
            return output
        

class BasicUNet_plus_cons_aug(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_cons_aug, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        self.cm = ContraModuleNew5(nb_filter[4])

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        if self.training:
            return output, self.cm(x4_0)
        else:
            return output

class BasicUNet_plus_mscn(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block_mscn, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_mscn, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        if self.training:
            return output, x4_0
        else:
            return output

class BasicUNet_plus_cons_cos(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(BasicUNet_plus_cons_cos, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        # self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        #     self.final5 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        self.cm = ContraModuleNew_2(nb_filter[4])

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))
        
        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_1)], 1))        
        xu1 = self.conv0_2_1x1(self.up4(x2_1))
        
        x1_1 = self.conv2_1(torch.cat([x1_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        # if self.deep_supervision:
        #     output1 = self.final1(xu3).sigmoid()
        #     output2 = self.final2(xu2).sigmoid()
        #     output3 = self.final3(xu1).sigmoid()
        #     output4 = self.final4(xu0).sigmoid()
        #     output5 = self.final5(xf).sigmoid()
        #     if self.training:
        #         return [output1, output2, output3, output4, output5]
        #     else:
        #         return output5
        # else:
        output = self.final(xf).sigmoid()
        if self.training:
            return output, self.cm(x4_0)
        else:
            return output