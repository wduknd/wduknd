import torch
import torch.nn as nn
import numpy as np

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
            avg_out = self.fc2(self.relu1(self.fc1(x.flatten(2).mean(dim=2, keepdim=True).unsqueeze(3))))  # for fast inference 这个和上面的结果不一致，不要轻易改变！
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

        window_kernel = np.asarray(k3x3)  # defaut 7x7
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