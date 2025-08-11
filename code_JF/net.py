from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from model import BasicUNet
from model.BasicUNet.model_BasicUNet import *
from utils import *
import os
from loss import *
from model import *
from focalloss import FocalLoss
# from model.pretrain.decoder import LightDecoder
# from model.pretrain.encoder import SparseEncoder
# from model.pretrain.models import build_sparse_encoder
# from model.pretrain.utils import arg_util, misc, lamb
# from skimage.feature.tests.test_orb import img

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        self.model_name = model_name
        
        self.cal_loss = SoftIoULoss()

        if model_name == 'DNANet':
            self.model = DNANet()  
        elif model_name == 'DNANet_MSCN':
            self.model = DNANet_MSCN()
        elif model_name == 'DNANet_MultiBranch':
            self.model = DNANet_MultiBranch()
        elif model_name == 'DNANet_MSCN_stage1':
            self.model = DNANet_MSCN_stage1()
        elif model_name == 'HintDNANet':
            self.model = HintDNANet()
        elif model_name == 'HintBasicUNet':
            self.model = HintBasicUNet()
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'ISNet':
            self.model = ISNet()
            self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            self.model = UIUNet()
        elif model_name == 'UIUNet_MSCN':
            self.model = UIUNet_MSCN()
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDUNet':
            self.model = ISTDU_Net()
        elif model_name == 'ISTDUNet-CLIP':
            self.model = ISTDU_Net_CLIP()
        elif model_name == 'ISTDU_Net_MSCN':
            self.model = ISTDU_Net_MSCN()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name == 'ResUNet':
            self.model = ResUNet()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        # elif model_name == 'MSHNet':
        #     self.model = MSHNet()
        #     self.cal_loss = SLSIoULoss()
        elif model_name == 'HomoFormer':
            self.model = HomoFormer()
        elif model_name == 'Uformer':
            self.model = Uformer()
        elif model_name == 'BasicUNet':
            self.model = BasicUNet()
        elif model_name == 'BasicUNet_plus_dropblock':
            self.model = BasicUNet_plus_dropblock()
        elif model_name == 'BasicUNet_plus_Spatialdrop':
            self.model = BasicUNet_plus_Spatialdrop()
        elif model_name == 'BasicUNet_woDeep':
            self.model = BasicUNet(deep_supervision=False)
        elif model_name == 'BasicUNet_simple':
            self.model = BasicUNet_Simple(deep_supervision=False)
        elif model_name == 'BasicUNet_plus':
            self.model = BasicUNet_plus()
            # self.min_loss = SimMinLoss()
            # self.max_loss = SimMaxLoss()            
            # self.cal_loss = SLSIoULoss()
        elif model_name == 'BasicUNet_plus_mscn':
            self.model = BasicUNet_plus_mscn()
        elif model_name == 'BasicUNet_plus_moe':
            self.model = BasicUNet_plus_moe()
        elif model_name == 'BasicUNet_plus_cons':
            self.model = BasicUNet_plus_cons()
            self.min_loss = SimMinLoss()
            self.max_loss = SimMaxLoss()            
            # self.cal_loss = SLSIoULoss()
        elif model_name =='BasicUNet_plus_cons_aug':
            self.model = BasicUNet_plus_cons_aug()
        elif model_name =='BasicUNet_plus_cons_cos':
            self.model = BasicUNet_plus_cons_cos()
        elif model_name == 'BasicUNet_plus_dysample':
            self.model = BasicUNet_plus_dysample()
            # self.cal_loss = SLSIoULoss()
        elif model_name == 'BasicUNet_plus2':
            self.model = BasicUNet_plus2()
        elif model_name == 'BasicUNet_pureRes':
            self.model = BasicUNet_pureRes()
        elif model_name == 'BasicUNet_fft':
            self.model = BasicUNet_fft()
        elif model_name =='BasicUNet_WOBOTH':
            self.model =BasicUNet_plus_woBOTH()
        elif model_name =='BasicUNet_WOCBAM':
            self.model =BasicUNet_plus_woCBAM()
        elif model_name =='BasicUNet_WOHFFM':
            self.model =BasicUNet_plus_woHFFM()
        elif model_name == 'HCFNet' :
            self.model = HCFNet()
            self.cal_loss = HCFNetLoss()
        elif model_name == 'MSHNet' :
            self.model = MSHNet(1)
        elif model_name == 'SCTransNet':
            self.model = SCTransNet()
            self.cal_loss = HierBCELoss()
            self.model.apply(weights_init_kaiming)
        elif model_name == 'VMUnet' :
            self.model = VMUNet()
        elif model_name =='MRF3Net':
            self.model = MRF3Net()
        elif model_name =='PBT':
            self.model = PBT()
        elif model_name =='RPCANet':
            self.model = RPCANet()
        elif model_name == 'VMUnet_CBAM' :
            self.model = VMUNet_CBAM()
        elif model_name == 'HintU':
            self.model = HintU()
            self.cal_loss =  FocalLoss()
        elif model_name == 'SeRankDet':
            self.model = SeRankDet()
        elif model_name  == 'L2SK_UNet':
            self.model = L2SKNet_UNet()
        # elif model_name  == 'Spark':
        #     # build encoder and decoder
        #     args: arg_util.Args = arg_util.init_dist_and_get_args()
        #     enc: SparseEncoder = build_sparse_encoder(args.model, input_size=args.input_size, sbn=args.sbn, drop_path_rate=args.dp, verbose=False)
        #     dec = LightDecoder(enc.downsample_raito, sbn=args.sbn)
        #     self.model = Spark( sparse_encoder=enc, dense_decoder=dec, 
        #                        mask_ratio=args.mask,densify_norm=args.densify_norm, sbn=args.sbn,)
        elif model_name == 'convnext':
            self.model = convnext()

    def weights_init_kaiming(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def forward(self, img, f=None, epoch=None):
        if f is not None:
            return self.model(img, f)
        elif epoch is not None:
            return self.model(img, epoch)
        else:
            return self.model(img)

    def loss(self, pred, gt_mask, epoch=None):
        if epoch is not None:
            loss = self.cal_loss(pred, gt_mask, epoch)
        else:
            loss = self.cal_loss(pred, gt_mask)
        return loss
    # def con_minloss(self, pred, gt_mask):
    #     loss = 
