from utils import *
import matplotlib.pyplot as plt
import os
import os.path as osp
import shutil
from PIL import Image, ImageOps, ImageFilter
from scipy.ndimage import uniform_filter
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class IRSTD_Dataset(Dataset):
    def __init__(self, dataset, params={}, mode='train', root='/home/pc/work/BasicIRSTD-main/data'):
        
        dataset_dir = osp.join(root, dataset)

        self.imgs_dir = osp.join(dataset_dir, 'images')
        self.label_dir = osp.join(dataset_dir, 'masks')
        self.names = []

        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset}.txt'), 'r') as f:
                self.names += [line.strip() for line in f.readlines()]
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset}.txt'), 'r') as f:
                self.names += [line.strip() for line in f.readlines()]
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset}.txt'), 'r') as f:
                self.names += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset}.txt'), 'r') as f:
                self.names += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        
        print(f'{len(self.names)} samples from {dataset} for {mode}')

        self.mode = mode
        self.crop_size = params.crop_size
        self.base_size = params.base_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._testval_sync_transform(img, mask)

        
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask

class DataSetLoader_Mixdecoder(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, mode, img_norm_cfg=None):
        super(DataSetLoader_Mixdecoder).__init__()
        self.dataset_name = dataset_name
        dataset_dir = osp.join(dataset_dir, dataset_name)
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size

        self.mode = mode

        self.files = []
        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]   # self.train_list = f.read().splitlines()
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        print(f'{len(self.files)} samples from {dataset_name} for {mode}')

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
    
    def generate_contrastive_mask(self, mask0, region_size=16, mask_ratio=0.2):  # mask0: [C, H, W]
        dh, dw = int(mask0.shape[0] // region_size), int(mask0.shape[0] // region_size)
        mask = np.zeros(dh * dw, dtype=np.float32)  
        mask[:round(dh * dw * mask_ratio)] = 1
        np.random.shuffle(mask)

        mask = mask.reshape(dh, dw)
        mask = mask.repeat(region_size, axis=0).repeat(region_size, axis=1)

        # 膨胀
        mask0 = nn.functional.max_pool2d(torch.from_numpy(mask0).unsqueeze(0), kernel_size=7, stride=1, padding=3).numpy()[0]

        mask[mask0 == 1] = 0  # positive 目标区域不进行mask
        return mask, mask0
    
    def __getitem__(self, idx):
        try:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.png')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.png'))
        except:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.bmp')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.bmp'))


        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
            
        if self.mode == 'train':
            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)
            
            img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

            filename = self.files[idx]
            if filename.startswith('XDU'):
                branch_id = 0
            elif filename.startswith('Misc'):
                branch_id = 1
            else:  # 默认纯数字
                branch_id = 2
            return img_patch, mask_patch,branch_id ,0

        else:
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            
            img, mask = img[np.newaxis,:], mask[np.newaxis,:]
            
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))
            if self.dataset_name == 'IRSTD-1K':
                branch_id = 0
            elif self.dataset_name == 'NUAA-SIRST':
                branch_id = 1
            elif self.dataset_name == 'NUDT-SIRST':
                branch_id = 2
            else:
                branch_id = -1  # 其它情况
            return img, mask, [h, w], self.files[idx], branch_id
    
    def __len__(self):
        return len(self.files)


class DataSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, mode, img_norm_cfg=None):
        super(DataSetLoader).__init__()
        self.dataset_name = dataset_name
        dataset_dir = osp.join(dataset_dir, dataset_name)
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size

        self.mode = mode

        self.files = []
        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]   # self.train_list = f.read().splitlines()
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        print(f'{len(self.files)} samples from {dataset_name} for {mode}')

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
    
    def generate_contrastive_mask(self, mask0, region_size=16, mask_ratio=0.2):  # mask0: [C, H, W]
        dh, dw = int(mask0.shape[0] // region_size), int(mask0.shape[0] // region_size)
        mask = np.zeros(dh * dw, dtype=np.float32)  
        mask[:round(dh * dw * mask_ratio)] = 1
        np.random.shuffle(mask)

        mask = mask.reshape(dh, dw)
        mask = mask.repeat(region_size, axis=0).repeat(region_size, axis=1)

        # 膨胀
        mask0 = nn.functional.max_pool2d(torch.from_numpy(mask0).unsqueeze(0), kernel_size=7, stride=1, padding=3).numpy()[0]

        mask[mask0 == 1] = 0  # positive 目标区域不进行mask
        return mask, mask0
    
    # def generate_contrastive_mask(self,mask0, region_size=16, mask_ratio=0.2):  # mask0: [C, H, W]
    #     # 获取 mask0 的形状
    #     H, W = mask0.shape
        
    #     # 计算 dh 和 dw，使用 np.ceil() 以处理非整除的情况
    #     dh = int(np.ceil(H / region_size))
    #     dw = int(np.ceil(W / region_size))

    #     mask = np.zeros(dh * dw, dtype=np.float32)
    #     mask[:round(dh * dw * mask_ratio)] = 1
    #     np.random.shuffle(mask)

    #     mask = mask.reshape(dh, dw)
    #     mask = np.repeat(mask, region_size, axis=0)[:H, :]
    #     mask = np.repeat(mask, region_size, axis=1)[:, :W]

    #     # 膨胀
    #     mask0 = nn.functional.max_pool2d(torch.from_numpy(mask0).unsqueeze(0), kernel_size=7, stride=1, padding=3).numpy()[0]

    #     mask[mask0 == 1] = 0   # positive 目标区域不进行mask
    #     return mask, mask0
    
    def __getitem__(self, idx):
        try:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.png')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.png'))
        except:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.bmp')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.bmp'))

        # print(self.files[idx])  # check for randomness, passed

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        # img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
            
        if self.mode == 'train':
            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)

            # # ====================================================================================
            # construct contrastive masking strategy
            # 1. 根据固定比例随机生成基于16*16 / 32*32 / 8*8的mask索引, 创建mask矩阵
            # 2. 构建正样本：将masking区域置为图像的均值, gt部分保留, label = label;
            # 3. 构建负样本：将gt区域膨胀5个像素点(基本充分掩盖目标), 将膨胀后的gt区域置为图像的均值, label = 0. 
            # if self.dataset_name == 'IRSTD-1K':
            #     mask_positive, mask_negative = self.generate_contrastive_mask(mask_patch.copy(), region_size=16, mask_ratio=0.2)
            # elif self.dataset_name == 'NUAA-SIRST':
            #     mask_positive, mask_negative = self.generate_contrastive_mask(mask_patch.copy(), region_size=16, mask_ratio=0.05)
            # else:    
            #     mask_positive, mask_negative = self.generate_contrastive_mask(mask_patch.copy(), region_size=48, mask_ratio=0.05)
    #         mask_positive, mask_negative = self.generate_contrastive_mask(mask_patch.copy(), region_size=16, mask_ratio=0.2)
    # # # === 计算局部块均值 ===
    #         region = 16
    #         h, w = img_patch.shape
    #         dh, dw = h // region, w // region

    #         # reshape + mean over patches
    #         patch_means = img_patch.reshape(dh, region, dw, region).mean(axis=(1, 3))  # shape: (dh, dw)

    #         # 恢复为原图大小，每个 patch 的均值填满16x16区域
    #         mean_full = np.repeat(np.repeat(patch_means, region, axis=0), region, axis=1)  # shape: (H, W)

    #         # === 构造正负样本 ===
    #         img_positive = img_patch.copy()
    #         img_positive[mask_positive == 1] = mean_full[mask_positive == 1]

    #         img_negative = img_patch.copy()
    #         img_negative[mask_negative == 1] = mean_full[mask_negative == 1]


    #         # fill_ = img_patch.mean()
    #         # fill_ = 0
    #         # img_positive = img_patch.copy()
    #         # img_positive[mask_positive == 1] = fill_
    #         # img_negative = img_patch.copy()
    #         # img_negative[mask_negative == 1] = fill_

    #         # import cv2
    #         # cv2.imwrite('img.png', img_patch.astype(np.uint8))
    #         # cv2.imwrite('img_pos.png', img_positive.astype(np.uint8))
    #         # cv2.imwrite('img_neg.png', img_negative.astype(np.uint8))

    #         img_positive, img_negative = img_positive[np.newaxis, :], img_negative[np.newaxis, :]
    #         img_positive = torch.from_numpy(np.ascontiguousarray(img_positive))
    #         img_negative = torch.from_numpy(np.ascontiguousarray(img_negative))

    #         # # # ====================================================================================
            
            img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

            return img_patch, mask_patch,0 ,0
            #return img_patch, mask_patch, img_positive, img_negative
        else:
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            
            img, mask = img[np.newaxis,:], mask[np.newaxis,:]
            
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))
            return img, mask, [h, w], self.files[idx]
    
    def __len__(self):
        return len(self.files)




class DataSetDomainLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, mode, img_norm_cfg=None):
        super(DataSetDomainLoader).__init__()
        self.dataset_name = dataset_name 
        dataset_dir = osp.join(dataset_dir, dataset_name)
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size

        self.mode = mode

        self.files = []
        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]   # self.train_list = f.read().splitlines()
                self.img_norm_cfg_train = get_img_norm_cfg(dataset_name, dataset_dir)
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
                self.img_norm_cfg_test = get_img_norm_cfg(dataset_name, dataset_dir)
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        print(f'{len(self.files)} samples from {dataset_name} for {mode}')

        # if img_norm_cfg == None:
        #     self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        # else:
        #     self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
    

    def __getitem__(self, idx):
        try:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.png')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.png'))
        except:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.bmp')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.bmp'))


        # img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        # mask = np.array(mask, dtype=np.float32)  / 255.0
        # if len(mask.shape) > 2:
        #     mask = mask[:,:,0]
            
        if self.mode == 'train':
            img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg_train)
            mask = np.array(mask, dtype=np.float32)  / 255.0
            if len(mask.shape) > 2:
                mask = mask[:,:,0]
            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)
            
            img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

            return img_patch, mask_patch,0 ,0
        
        else:
            img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg_test)
            mask = np.array(mask, dtype=np.float32)  / 255.0
            if len(mask.shape) > 2:
                mask = mask[:,:,0]
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            
            img, mask = img[np.newaxis,:], mask[np.newaxis,:]
            
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))
            return img, mask, [h, w], self.files[idx]
    
    def __len__(self):
        return len(self.files)



import math
class MixedDomainBatchLoader(Dataset):
    def __init__(self, syn_dataset, real_dataset, syn_ratio=0.75, batchsize=16):
        self.syn_dataset = syn_dataset
        self.real_dataset = real_dataset
        self.syn_ratio = syn_ratio
        self.real_ratio = 1.0 - syn_ratio

        self.syn_len = len(syn_dataset)
        self.real_len = len(real_dataset)
        self.batch_size = batchsize

        # 每 batch 中合成/真实样本数
        self.b_syn = int(self.syn_ratio * self.batch_size)
        self.b_real = self.batch_size - self.b_syn

        # 保证每类样本至少用一遍，计算总 batch 数
        self.num_batches = max(
            math.ceil(self.syn_len / self.b_syn),
            math.ceil(self.real_len / self.b_real)
        )
        self.total_len = self.num_batches * self.batch_size

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        batch_position = idx % self.batch_size
        if batch_position < self.b_syn:
            sample_idx = (idx // self.batch_size * self.b_syn + batch_position) % self.syn_len
            img, mask, _, _ = self.syn_dataset[sample_idx]
            domain_label = 0
        else:
            sample_idx = (idx // self.batch_size * self.b_real + (batch_position - self.b_syn)) % self.real_len
            img, mask, _, _ = self.real_dataset[sample_idx]
            domain_label = 1

        return img, mask, domain_label

class BalancedPairedDomainDataset(Dataset):
    def __init__(self, syn_dataset, real_dataset):
        """
        合成和真实数据按1:1配对加载。
        较小数据集将被重复采样以匹配大数据集。
        """
        assert len(syn_dataset) > 0 and len(real_dataset) > 0, "两个数据集都不能为空！"
        self.syn_dataset = syn_dataset
        self.real_dataset = real_dataset

        self.syn_len = len(syn_dataset)
        self.real_len = len(real_dataset)

        self.dataset_len = max(self.syn_len, self.real_len)

        self.set_epoch(0)

    def set_epoch(self, epoch):
        random.seed(epoch)
        # 确定长短
        if self.syn_len >= self.real_len:
            self.long_dataset = self.syn_dataset
            self.short_dataset = self.real_dataset
            self.long_indices = list(range(self.syn_len))  # 不重复
            self.short_indices = self._build_balanced_indices(self.real_len, self.syn_len)
        else:
            self.long_dataset = self.real_dataset
            self.short_dataset = self.syn_dataset
            self.long_indices = list(range(self.real_len))
            self.short_indices = self._build_balanced_indices(self.syn_len, self.real_len)

        # 打乱长短索引顺序
        random.shuffle(self.long_indices)
        random.shuffle(self.short_indices)

    def _build_balanced_indices(self, short_len, target_len):
        # 每个样本至少重复 floor(n) 次
        base_repeats = target_len // short_len
        remainder = target_len % short_len

        indices = list(range(short_len)) * base_repeats
        indices += random.sample(range(short_len), remainder)
        return indices

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        long_item = self.long_dataset[self.long_indices[idx]]
        short_item = self.short_dataset[self.short_indices[idx]]

        # 按原始配对顺序返回
        if self.syn_len >= self.real_len:
            syn_img, syn_mask, *_ = long_item
            real_img, real_mask, *_ = short_item
        else:
            syn_img, syn_mask, *_ = short_item
            real_img, real_mask, *_ = long_item

        return {
            'syn_img': syn_img,
            'syn_mask': syn_mask,
            'real_img': real_img,
            'real_mask': real_mask,
        }
    

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//','/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//','/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        
        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

class InferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(InferenceSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        h, w = img.shape
        img = PadImg(img)
        
        img = img[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 


class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png').replace('//','/'))
        mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
        
        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:,:,0]
        
        h, w = mask_pred.shape
        
        mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h,w]
    def __len__(self):
        return len(self.test_list) 

# class EvalSetLoader(Dataset):
#     def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name):
#         super(EvalSetLoader).__init__()
#         self.dataset_dir = '/home/pc/work/BasicIRSTD-main/data'
#         # self.mask_pred_dir = '/home/pc/work/IRSTD-Ablation/BasicIRSTD-main'
#         self.mask_pred_dir = '/home/pc/work/IRSTD-Ablation/BasicIRSTD-main/Traditional_methods/NRAM-master/Demo_NRAM/results'
#         self.test_dataset_name = test_dataset_name

#         with open(self.dataset_dir+'/'+test_dataset_name+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
#             self.test_list = f.read().splitlines()

#     def __getitem__(self, idx):
#         # mask_pred = Image.open((self.mask_pred_dir + '/'+ self.test_dataset_name + '/' +'IPI'+'/'+'E' + '/' + self.test_list[idx] + '.png').replace('//','/'))
#         mask_pred = Image.open((self.mask_pred_dir + '/'+ self.test_dataset_name + '/' + 'target/' + self.test_list[idx] + '.png').replace('//','/'))
#         mask_gt = Image.open(self.dataset_dir +'/'+self.test_dataset_name + '/masks/' + self.test_list[idx] + '.png')

#         mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
#         mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
#         if len(mask_gt.shape) > 2:
#             mask_gt = mask_gt[:,:,0]
#         if len(mask_pred.shape) == 3:
#             mask_pred = mask_pred[:,:,0]
#         # if len(mask_gt.shape) == 5:
#         #     print('1')
        
#         h, w = mask_pred.shape
        
#         mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
#         mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
#         mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
#         # if len(mask_gt.shape) == 5:
#         #     print('1')
#         # if idx == 90:
#         #     print('1')
#         return mask_pred, mask_gt, [h,w]
#     def __len__(self):
#         return len(self.test_list) 

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]

            # print('aug: 1') # check for randomness, passed
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
            # print('aug: 2') # check for randomness, passed
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
            # print('aug: 3') # check for randomness, passed
        return input, target
