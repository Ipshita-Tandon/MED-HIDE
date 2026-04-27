# psnr in 40s and passes 2 robustness tests [final code!]

# ==========================================
# FINAL STEGO GAN - MILD ROBUSTNESS & HIGH PSNR
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF 
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.amp as amp 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from tqdm import tqdm
import gc
import io
import random
import zlib
from reedsolo import RSCodec

# ==========================================
# FIXED CHAOTIC POOL (YOUR MASTER KEY)
# ==========================================
import torch

CHAOTIC_POOL = []

# 1. Lock the random seed so the permutations are identical across server restarts
torch.manual_seed(42) # "42" is your master password. Don't change it!

# 2. Generate the deterministic pool
for _ in range(10):
    idx = torch.randperm(256 * 256)
    inv = torch.argsort(idx)
    CHAOTIC_POOL.append({'idx': idx, 'inv': inv})

# 3. Release the seed so the rest of your app remains truly random
torch.seed()

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    lr_G = 2e-4             
    lr_D = 1e-4             
    batch_size = 4          
    epochs = 20             
    cropsize = 256
    
    lam_adv = 1.0           
    lam_image = 250.0       
    lam_data = 50.0         
    lam_percep = 20.0       
    noise_std = 0.01        
    
    DATA_PATHS = [
        '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset',
        '/kaggle/input/datasets/pcbreviglieri/pneumonia-xray-images',
        '/kaggle/input/datasets/kmader/siim-medical-images/dicom_dir'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

c = Config()

# ==========================================
# 2. UTILS & LAYERS
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, ch, _, _ = x.size()
        y = self.avg_pool(x).view(b, ch)
        y = self.fc(y).view(b, ch, 1, 1)
        return x * y.expand_as(x)

class JpegBasic(nn.Module):
	def __init__(self):
		super(JpegBasic, self).__init__()

	def std_quantization(self, image_yuv_dct, scale_factor, round_func=torch.round):

		luminance_quant_tbl = (torch.tensor([
			[16, 11, 10, 16, 24, 40, 51, 61],
			[12, 12, 14, 19, 26, 58, 60, 55],
			[14, 13, 16, 24, 40, 57, 69, 56],
			[14, 17, 22, 29, 51, 87, 80, 62],
			[18, 22, 37, 56, 68, 109, 103, 77],
			[24, 35, 55, 64, 81, 104, 113, 92],
			[49, 64, 78, 87, 103, 121, 120, 101],
			[72, 92, 95, 98, 112, 100, 103, 99]
		], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
			image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

		chrominance_quant_tbl = (torch.tensor([
			[17, 18, 24, 47, 99, 99, 99, 99],
			[18, 21, 26, 66, 99, 99, 99, 99],
			[24, 26, 56, 99, 99, 99, 99, 99],
			[47, 66, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99]
		], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
			image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

		q_image_yuv_dct = image_yuv_dct.clone()
		q_image_yuv_dct[:, :1, :, :] = image_yuv_dct[:, :1, :, :] / luminance_quant_tbl
		q_image_yuv_dct[:, 1:, :, :] = image_yuv_dct[:, 1:, :, :] / chrominance_quant_tbl
		q_image_yuv_dct_round = round_func(q_image_yuv_dct)
		return q_image_yuv_dct_round

	def std_reverse_quantization(self, q_image_yuv_dct, scale_factor):

		luminance_quant_tbl = (torch.tensor([
			[16, 11, 10, 16, 24, 40, 51, 61],
			[12, 12, 14, 19, 26, 58, 60, 55],
			[14, 13, 16, 24, 40, 57, 69, 56],
			[14, 17, 22, 29, 51, 87, 80, 62],
			[18, 22, 37, 56, 68, 109, 103, 77],
			[24, 35, 55, 64, 81, 104, 113, 92],
			[49, 64, 78, 87, 103, 121, 120, 101],
			[72, 92, 95, 98, 112, 100, 103, 99]
		], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
			q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

		chrominance_quant_tbl = (torch.tensor([
			[17, 18, 24, 47, 99, 99, 99, 99],
			[18, 21, 26, 66, 99, 99, 99, 99],
			[24, 26, 56, 99, 99, 99, 99, 99],
			[47, 66, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99]
		], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
			q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

		image_yuv_dct = q_image_yuv_dct.clone()
		image_yuv_dct[:, :1, :, :] = q_image_yuv_dct[:, :1, :, :] * luminance_quant_tbl
		image_yuv_dct[:, 1:, :, :] = q_image_yuv_dct[:, 1:, :, :] * chrominance_quant_tbl
		return image_yuv_dct

	def dct(self, image):
		# coff for dct and idct
		coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
		coff[0, :] = 1 * np.sqrt(1 / 8)
		for i in range(1, 8):
			for j in range(8):
				coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

		split_num = image.shape[2] // 8
		image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
		image_dct = torch.matmul(coff, image_dct)
		image_dct = torch.matmul(image_dct, coff.permute(1, 0))
		image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

		return image_dct

	def idct(self, image_dct):
		# coff for dct and idct
		coff = torch.zeros((8, 8), dtype=torch.float).to(image_dct.device)
		coff[0, :] = 1 * np.sqrt(1 / 8)
		for i in range(1, 8):
			for j in range(8):
				coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

		split_num = image_dct.shape[2] // 8
		image = torch.cat(torch.cat(image_dct.split(8, 2), 0).split(8, 3), 0)
		image = torch.matmul(coff.permute(1, 0), image)
		image = torch.matmul(image, coff)
		image = torch.cat(torch.cat(image.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

		return image

	def rgb2yuv(self, image_rgb):
		image_yuv = torch.empty_like(image_rgb)
		image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] \
								  + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
		image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] \
								  - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
		image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] \
								  - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
		return image_yuv

	def yuv2rgb(self, image_yuv):
		image_rgb = torch.empty_like(image_yuv)
		image_rgb[:, 0:1, :, :] = image_yuv[:, 0:1, :, :] + 1.40198758 * image_yuv[:, 2:3, :, :]
		image_rgb[:, 1:2, :, :] = image_yuv[:, 0:1, :, :] - 0.344113281 * image_yuv[:, 1:2, :, :] \
								  - 0.714103821 * image_yuv[:, 2:3, :, :]
		image_rgb[:, 2:3, :, :] = image_yuv[:, 0:1, :, :] + 1.77197812 * image_yuv[:, 1:2, :, :]
		return image_rgb

	def yuv_dct(self, image, subsample):
		# clamp and convert from [-1,1] to [0,255]
		image = (image.clamp(-1, 1) + 1) * 255 / 2

		# pad the image so that we can do dct on 8x8 blocks
		pad_height = (8 - image.shape[2] % 8) % 8
		pad_width = (8 - image.shape[3] % 8) % 8
		image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image)

		# convert to yuv
		image_yuv = self.rgb2yuv(image)

		assert image_yuv.shape[2] % 8 == 0
		assert image_yuv.shape[3] % 8 == 0

		# subsample
		image_subsample = self.subsampling(image_yuv, subsample)

		# apply dct
		image_dct = self.dct(image_subsample)

		return image_dct, pad_width, pad_height

	def idct_rgb(self, image_quantization, pad_width, pad_height):
		# apply inverse dct (idct)
		image_idct = self.idct(image_quantization)

		# transform from yuv to to rgb
		image_ret_padded = self.yuv2rgb(image_idct)

		# un-pad
		image_rgb = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
					:image_ret_padded.shape[3] - pad_width].clone()

		return image_rgb * 2 / 255 - 1

	def subsampling(self, image, subsample):
		if subsample == 2:
			split_num = image.shape[2] // 8
			image_block = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
			for i in range(8):
				if i % 2 == 1: image_block[:, 1:3, i, :] = image_block[:, 1:3, i - 1, :]
			for j in range(8):
				if j % 2 == 1: image_block[:, 1:3, :, j] = image_block[:, 1:3, :, j - 1]
			image = torch.cat(torch.cat(image_block.chunk(split_num, 0), 3).chunk(split_num, 0), 2)
		return image


class JpegSS(JpegBasic):
	def __init__(self, Q, subsample=0):
		super(JpegSS, self).__init__()

		# quantization table
		self.Q = Q
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		# subsample
		self.subsample = subsample

	def round_ss(self, x):
		cond = torch.tensor((torch.abs(x) < 0.5), dtype=torch.float).to(x.device)
		return cond * (x ** 3) + (1 - cond) * x

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# quantization
		image_quantization = self.std_quantization(image_dct, self.scale_factor, self.round_ss)

		# reverse quantization
		image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
		return noised_image.clamp(-1, 1)

class SP(nn.Module):

	def __init__(self, prob):
		super(SP, self).__init__()
		self.prob = prob

	def sp_noise(self, image, prob):
		prob_zero = prob / 2
		prob_one = 1 - prob_zero
		rdn = torch.rand(image.shape).to(image.device)

		output = torch.where(rdn > prob_one, torch.zeros_like(image).to(image.device), image)
		output = torch.where(rdn < prob_zero, torch.ones_like(output).to(output.device), output)

		return output

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		return self.sp_noise(image, self.prob)

class RobustnessLayer(nn.Module):
    def __init__(self, std=0.015):
        super().__init__()
        self.std = std
        self.diff_jpeg = JpegSS(Q=90)
        self.sp_noise_layer = SP(prob=0.02) 

    def forward(self, x, epoch):
        if not self.training:
            return x

        x = x + torch.randn_like(x) * self.std
        x_quantized = torch.round(x * 255.0) / 255.0
        x = x + (x_quantized - x).detach()
        x = torch.clamp(x, 0, 1)

        if epoch < 3: # Shortened warmup to get into the fight faster
            return x

        attack_type = random.choice([
            'none', 'jpeg_diff', 'blur', 'sp_noise', 'gaussian_noise', 'spatial_dropout'
        ])

        if attack_type == 'jpeg_diff':
            q = random.randint(85, 95) 
            self.diff_jpeg.Q = q
            self.diff_jpeg.scale_factor = 2 - q * 0.02 if q >= 50 else 50 / q
            x = self.diff_jpeg((x, x))
            
        elif attack_type == 'sp_noise':
            self.sp_noise_layer.prob = random.uniform(0.01, 0.02)
            x = self.sp_noise_layer((x, x))
            
        elif attack_type == 'blur':
            x = TF.gaussian_blur(x, kernel_size=3, sigma=[0.5, 1.0])
            
        elif attack_type == 'gaussian_noise':
            # Train on the failing sigma=0.05 early, calm down later for PSNR
            current_sigma = random.uniform(0.03, 0.055) if epoch < 12 else random.uniform(0.01, 0.025)
            noise_heavy = torch.randn_like(x) * current_sigma
            x = torch.clamp(x + noise_heavy, 0, 1)
            
        elif attack_type == 'spatial_dropout':
            # Simulate the 5% benchmark dropout (pixels turning black)
            drop_prob = random.uniform(0.04, 0.07) if epoch < 12 else random.uniform(0.01, 0.02)
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) > drop_prob).float()
            x = x * mask

        x_quantized = torch.round(x * 255.0) / 255.0
        x = x + (x_quantized - x).detach()
        return torch.clamp(x, 0, 1)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        x_norm = (x - self.mean) / self.std
        y_norm = (y - self.mean) / self.std
        with torch.no_grad():
            feat_x = self.features(x_norm)
            feat_y = self.features(y_norm)
        return F.mse_loss(feat_x, feat_y)

# ==========================================
# 3. MODELS (GENERATOR & DISCRIMINATOR)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c)
        )
        self.se = SEBlock(in_c)

    def forward(self, x):
        return x + self.se(self.conv(x))

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class StegoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = HaarDWT()
        self.iwt = HaarIWT()
        
        self.down1 = ConvBlock(16, 64)
        self.pool1 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.down2 = ConvBlock(64, 128)
        self.pool2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.down3 = ConvBlock(128, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(128, 64) 
        
        self.residual_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 12, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.attention_head = nn.Sequential(
            nn.Conv2d(64, 12, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        # FIX: Parameter allows dynamic clamping and optimizer control
        self.base_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, img, data):
        img_dwt = self.dwt(img)    
        data_dwt = self.dwt(data)  
        x = torch.cat([img_dwt, data_dwt], dim=1) 
        
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        
        u1 = self.up1(d3)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up_conv1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        out_features = self.up_conv2(u2)
        
        raw_residual = self.residual_head(out_features)
        attention_map = self.attention_head(out_features)
        
        final_residual_dwt = raw_residual * attention_map * self.base_scale
        
        # --- NEW: Suppress LL Band (Channels 0, 1, 2) ---
        # Forces data into high-frequency bands, massively boosting PSNR
        mask_ll = torch.ones_like(final_residual_dwt)
        mask_ll[:, :3, :, :] = 0.1 
        final_residual_dwt = final_residual_dwt * mask_ll
        
        stego_dwt = img_dwt + final_residual_dwt
        
        stego_img = self.iwt(stego_dwt)
        stego_img = torch.clamp(stego_img, 0, 1)
        
        return stego_img, attention_map

class StegoDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = HaarDWT()
        self.iwt = HaarIWT()
        
        self.head = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dilated1 = nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2)
        self.dilated2 = nn.Conv2d(64, 32, kernel_size=3, padding=4, dilation=4)
        self.dilated3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, dilation=1)
        self.body = nn.Sequential(
            ResBlock(160), ResBlock(160), ResBlock(160), ResBlock(160), ResBlock(160)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),      
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 4, kernel_size=3, padding=1) 
        )

    def forward(self, x):
        x_dwt = self.dwt(x)
        features = self.head(x_dwt)
        d1 = F.leaky_relu(self.dilated1(features), 0.2)
        d2 = F.leaky_relu(self.dilated2(features), 0.2)
        d3 = F.leaky_relu(self.dilated3(features), 0.2)
        multi_scale = torch.cat([features, d1, d2, d3], dim=1)
        
        processed = self.body(multi_scale)
        pred_data_dwt = self.tail(processed)
        return self.iwt(pred_data_dwt)

class SpatialAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, 1, 1) 
        )
    def forward(self, x): return self.model(x)

class TransformDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(12, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, 1, 1) 
        )
    def forward(self, x): return self.model(x)

class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filters = torch.stack([ll, hl, lh, hh]).unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b*c, 1, h, w)
        out = F.conv2d(x, self.filters, stride=2) 
        out = out.view(b, c, 4, h//2, w//2)
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(b, c*4, h//2, w//2)
        return out

class HaarIWT(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        filters = torch.stack([ll, hl, lh, hh]).unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, x):
        b, c4, h_half, w_half = x.shape
        c = c4 // 4
        x = x.view(b, 4, c, h_half, w_half)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b*c, 4, h_half, w_half)
        out = F.conv_transpose2d(x, self.filters, stride=2) 
        out = out.view(b, c, h_half*2, w_half*2)
        return out

# ==========================================
# 4. UPDATED PAYLOAD ENGINE (Original Stable Version)
# ==========================================
class PayloadEngine:
    HEADER_LEN = 32
    rs = RSCodec(32) 

    @staticmethod
    def str_to_bits(text):
        try:
            data_bytes = zlib.compress(text.encode('utf-8'))
            chunk_size = 100
            encoded_bits = ""
            for i in range(0, len(data_bytes), chunk_size):
                chunk = data_bytes[i:i+chunk_size]
                ecc_chunk = PayloadEngine.rs.encode(bytearray(chunk))
                encoded_bits += ''.join(format(b, '08b') for b in ecc_chunk)
            return encoded_bits
        except Exception as e:
            print(f"⚠️ [ENCODING ERROR]: {e}") 
            return ""

    @staticmethod
    def decode(tensor):
        flat = (tensor.view(-1) > 0.5).int().cpu().numpy()
        bit_str = "".join(str(b) for b in flat)
        try:
            header_reps = 15
            header_len = PayloadEngine.HEADER_LEN
            header_raw = bit_str[:header_len * header_reps]
            header_votes = [0] * header_len
            for i in range(header_reps):
                for j in range(header_len):
                    header_votes[j] += int(header_raw[i * header_len + j])
            
            payload_len_bin = "".join(['1' if v > (header_reps // 2) else '0' for v in header_votes])
            payload_len = int(payload_len_bin, 2)
            
            if payload_len <= 0 or payload_len > len(bit_str): 
                return f"ERR_LEN (Parsed Length: {payload_len})"
            
            start_idx = header_len * header_reps
            block_size_bits = (100 + 32) * 8
            full_data = bit_str[start_idx : start_idx + payload_len]
            
            decoded_bytes = bytearray()
            for i in range(0, len(full_data), block_size_bits):
                block = full_data[i : i + block_size_bits]
                if len(block) < 16: break
                byte_list = [int(block[j:j+8], 2) for j in range(0, len(block), 8) if len(block[j:j+8]) == 8]
                try:
                    decoded_chunk = PayloadEngine.rs.decode(bytearray(byte_list))[0]
                    decoded_bytes.extend(decoded_chunk)
                except:
                    return "[CHUNK_LOST_ECC]"
            
            try:
                return zlib.decompress(decoded_bytes).decode('utf-8')
            except zlib.error as e:
                return f"[ZLIB_DECOMPRESSION_FAILED: {str(e)}]"
                
        except Exception as e:
            return f"FATAL_ECC: {str(e)}"
    
    @staticmethod
    def fast_scramble(tensor, pool_idx=None):
        b, c_in, h, w = tensor.shape
        if pool_idx is None: pool_idx = random.randint(0, 9)
        key_data = CHAOTIC_POOL[pool_idx]
        flat = tensor.view(b, -1)
        scrambled = flat[:, key_data['idx']].view(b, c_in, h, w)
        return scrambled, pool_idx

    @staticmethod
    def fast_descramble(tensor, pool_idx):
        b, c_in, h, w = tensor.shape
        key_data = CHAOTIC_POOL[pool_idx]
        flat = tensor.view(b, -1)
        descrambled = flat[:, key_data['inv']].view(b, c_in, h, w)
        return descrambled

    @staticmethod
    def generate_random_batch(batch_size, img_size=256, bpp=1.0):
        capacity = img_size * img_size
        num_bits = int(capacity * bpp)
        payload = torch.full((batch_size, capacity), -1.0)
        if num_bits > 0:
            payload[:, :num_bits] = torch.bernoulli(torch.full((batch_size, num_bits), 0.5))
        return payload.view(batch_size, 1, img_size, img_size)

# ==========================================
# 5. DATA LOADING & TRAINING LOOP
# ==========================================
class UniversalDataset(Dataset):
    def __init__(self, root_dirs, cropsize=256):
        self.files = []
        extensions = ['*.jpg', '*.jpeg', '*.dcm', '*.png']
        for root in root_dirs:
            for ext in extensions:
                self.files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        
        if len(self.files) == 0:
            print("⚠️ No images found. Using Dummy Data for Testing.")
            self.files = ["dummy" for _ in range(100)]
            self.dummy = True
        else:
            self.dummy = False

        self.tf = T.Compose([
            T.Resize((cropsize, cropsize)),
            T.ToTensor()
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        if self.dummy: return torch.rand(3, c.cropsize, c.cropsize)
        path = self.files[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.tf(img)
        except:
            return torch.zeros(3, c.cropsize, c.cropsize)

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0: return 100
    return 10 * torch.log10(1.0 / mse)

def rgb_to_yuv(img):
    yuv = torch.empty_like(img)
    yuv[:, 0, :, :] =  0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
    yuv[:, 1, :, :] = -0.147 * img[:, 0, :, :] - 0.289 * img[:, 1, :, :] + 0.436 * img[:, 2, :, :]
    yuv[:, 2, :, :] =  0.615 * img[:, 0, :, :] - 0.515 * img[:, 1, :, :] - 0.100 * img[:, 2, :, :]
    return yuv

def ssim_loss(x, y, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, window_size, 1, window_size//2)
    mu_y = F.avg_pool2d(y, window_size, 1, window_size//2)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, window_size//2) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, window_size, 1, window_size//2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, window_size//2) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim_map.mean()

def train_gan():
    full_ds = UniversalDataset(c.DATA_PATHS, cropsize=c.cropsize)
    images_per_epoch = 500 
    indices = torch.randperm(len(full_ds))[:images_per_epoch]
    ds = torch.utils.data.Subset(full_ds, indices)
    dl = DataLoader(ds, batch_size=c.batch_size, shuffle=True, drop_last=True)
    
    enc = StegoEncoder().to(c.device)
    dec = StegoDecoder().to(c.device)
    
    analyzer_A = SpatialAnalyzer().to(c.device)
    disc_D = TransformDiscriminator().to(c.device)
    
    noise_layer = RobustnessLayer(c.noise_std).to(c.device)
    try: vgg_loss = PerceptualLoss().to(c.device)
    except: vgg_loss = None

    opt_G = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=c.lr_G, betas=(0.5, 0.999))
    opt_A = optim.Adam(analyzer_A.parameters(), lr=c.lr_D, betas=(0.5, 0.999))
    opt_D = optim.Adam(disc_D.parameters(), lr=c.lr_D, betas=(0.5, 0.999))
    
    scaler = amp.GradScaler('cuda')
    mse_loss = nn.MSELoss()

    for epoch in range(1, c.epochs + 1):
        
        # --- STABLE SCHEDULER ---
        if epoch == 1:
            enc.base_scale.data = torch.tensor(0.06).to(c.device)
            c.lam_data = 150.0 
        elif epoch == 4:
            enc.base_scale.data = torch.tensor(0.05).to(c.device)
            c.lam_data = 200.0 # Massive boost to fight the new heavy noise 
        elif epoch == 12:
            print("\n📉 Initiating Polish Phase: Prioritizing Noise Survival...")
            for param_group in opt_G.param_groups: param_group['lr'] = 5e-5
            for param_group in opt_A.param_groups: param_group['lr'] = 2e-5
            for param_group in opt_D.param_groups: param_group['lr'] = 2e-5
            
            # Start the polish at a much safer, louder amplitude
            enc.base_scale.data = torch.tensor(0.045).to(c.device)
            
            # Keep data priority HIGHER than image priority so it doesn't sacrifice bits
            c.lam_image = 150.0 
            c.lam_data = 250.0

        enc.train(); dec.train(); analyzer_A.train(); disc_D.train()
        pbar = tqdm(dl, leave=False)
        metrics = {'g_loss': [], 'psnr': [], 'ber': []}
        
        for cover in pbar:
            current_std = min(c.noise_std, c.noise_std * (epoch / 10))
            noise_layer.std = current_std 
            cover = cover.to(c.device)
            batch_dim = cover.size(0)
            
            # --- CAPPED CURRICULUM TO PREVENT BACKGROUND NOISE ---
            if epoch < 5: 
                current_bpp = random.choice([0.1, 0.25]) 
            elif epoch < 10: 
                current_bpp = random.choice([0.25, 0.4])
            else: 
                current_bpp = random.choice([0.25, 0.4, 0.5]) # Capped at 0.5
                
            secret = PayloadEngine.generate_random_batch(batch_dim, bpp=current_bpp).to(c.device)
            scrambled_secret, p_idx = PayloadEngine.fast_scramble(secret)

            # ========================================================
            # STEP 1: GENERATE STEGO ONCE (Forward Pass)
            # ========================================================
            with amp.autocast('cuda'):
                stego, attention_map = enc(cover, scrambled_secret)
                stego_dwt = enc.dwt(stego)
                cover_dwt = enc.dwt(cover)

            # ========================================================
            # STEP 2: UPDATE ADVERSARIES (Using Detached Stego)
            # ========================================================
            opt_A.zero_grad()
            with amp.autocast('cuda'):
                pred_real_A = analyzer_A(cover)
                pred_fake_A = analyzer_A(stego.detach()) 
                loss_A = (mse_loss(pred_real_A, torch.ones_like(pred_real_A)) + 
                          mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A))) * 0.5
            scaler.scale(loss_A).backward()
            scaler.step(opt_A)

            opt_D.zero_grad()
            with amp.autocast('cuda'):
                pred_real_D = disc_D(cover_dwt)
                pred_fake_D = disc_D(stego_dwt.detach()) 
                loss_D = (mse_loss(pred_real_D, torch.ones_like(pred_real_D)) + 
                          mse_loss(pred_fake_D, torch.zeros_like(pred_fake_D))) * 0.5
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)

            # ========================================================
            # STEP 3: UPDATE GENERATOR
            # ========================================================
            opt_G.zero_grad()
            with amp.autocast('cuda'):
                loss_adv_A = mse_loss(analyzer_A(stego), torch.ones_like(pred_real_A))
                loss_adv_D = mse_loss(disc_D(stego_dwt), torch.ones_like(pred_real_D))
                
                # --- EDGE & BACKGROUND AWARENESS ---
                # Detect edges (where it is safe to hide data)
                mask_h = torch.abs(cover[:, :, 1:, :] - cover[:, :, :-1, :])
                mask_w = torch.abs(cover[:, :, :, 1:] - cover[:, :, :, :-1])
                mask = F.pad(mask_h, (0, 0, 0, 1)) + F.pad(mask_w, (0, 1, 0, 0))
                edge_mask = (mask > 0.05).float() 
                smooth_mask = 1.0 - edge_mask # Areas that are flat/smooth
                
                # Strict background detection (pure black areas)
                background_mask = (torch.mean(cover, dim=1, keepdim=True) < 0.05).float()
                
                residual = stego - cover
                loss_img_mse = mse_loss(stego, cover)
                loss_img_l1 = torch.nn.functional.l1_loss(stego, cover)
                loss_linf = torch.max(torch.abs(residual))
                loss_img_vgg = vgg_loss(stego, cover) if vgg_loss else 0.0
                
                # New Aggressive Spatial Penalties
                loss_smooth = torch.mean(torch.abs(residual) * smooth_mask)
                loss_bg_strict = torch.mean(torch.abs(residual) * background_mask)
                
                noisy_stego = noise_layer(stego, epoch)
                recovered_logits = dec(noisy_stego)
                
                valid_mask = (scrambled_secret != -1.0).float()
                target_bits = torch.clamp(scrambled_secret, 0.0, 1.0)
                
                bce = F.binary_cross_entropy_with_logits(recovered_logits, target_bits, reduction='none')
                with torch.no_grad():
                    probs = torch.sigmoid(recovered_logits)
                    error_dist = torch.abs(probs - target_bits)
                    # Boosted from 3.0 to 6.0 to violently punish bit corruption from dropout
                    focal_weight = 1.0 + (error_dist * 6.0) 
                
                loss_data = (bce * focal_weight * valid_mask).sum() / (valid_mask.sum() + 1e-8)

                cover_yuv = rgb_to_yuv(cover)
                stego_yuv = rgb_to_yuv(stego)
                # Punish color shifts heavily (medical images shouldn't have random color noise)
                loss_color_preservation = mse_loss(stego_yuv[:, 0], cover_yuv[:, 0]) + (15.0 * mse_loss(stego_yuv[:, 1:], cover_yuv[:, 1:]))

                residual = stego - cover
                background_mask = (cover < 0.05).float() 
                tissue_mask = 1.0 - background_mask
                loss_bg = torch.mean(torch.abs(residual) * background_mask)
                loss_tissue = torch.mean(torch.abs(residual) * tissue_mask)
                
                # --- FREQUENCY DOMAIN PENALTIES ---
                loss_freq_ll = torch.mean(attention_map[:, :3, :, :]) 
                loss_freq_mid = torch.mean(attention_map[:, 3:9, :, :]) 
                loss_freq_hh = torch.mean(attention_map[:, 9:, :, :])
                
                loss_sparsity = torch.mean(attention_map)

                # --- THE "DIAMOND PSNR" SCHEDULER ---
                if epoch < 5:
                    curr_lam_data = 250.0 
                    curr_lam_image = 100.0 
                    l1_weight = 50.0
                    linf_weight = 10.0
                    ssim_w = 10.0
                    bg_weight = 50.0
                    smooth_weight = 10.0
                    color_weight = 50.0
                elif epoch < 12:
                    curr_lam_data = 150.0
                    curr_lam_image = 300.0 
                    l1_weight = 150.0
                    linf_weight = 50.0
                    ssim_w = 40.0
                    bg_weight = 200.0
                    smooth_weight = 50.0
                    color_weight = 150.0
                else:
                    curr_lam_data = 150.0  # Increased from 80 to maintain dropout memory
                    curr_lam_image = 800.0 
                    l1_weight = 500.0      
                    linf_weight = 200.0    
                    ssim_w = 150.0  
                    bg_weight = 1000.0     
                    smooth_weight = 300.0  
                    color_weight = 500.0

                # The Ultimate Loss Function
                loss_G = (c.lam_adv * (loss_adv_A + loss_adv_D)) + \
                         (curr_lam_image * loss_img_mse) + \
                         (l1_weight * loss_img_l1) + \
                         (linf_weight * loss_linf) + \
                         (c.lam_percep * loss_img_vgg) + \
                         (curr_lam_data * loss_data) + \
                         (ssim_w * ssim_loss(stego, cover)) + \
                         (bg_weight * loss_bg_strict) + \
                         (smooth_weight * loss_smooth) + \
                         (color_weight * loss_color_preservation)
            
            scaler.scale(loss_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1.0)
            scaler.step(opt_G)
            scaler.update()
            
            # THE HARD FLOOR: Lowered to unlock 41+ dB
            with torch.no_grad():
                if epoch < 12:
                    enc.base_scale.data.clamp_(min=0.01, max=0.05)
                else:
                    # Dropped to 0.001 to allow absolute minimal pixel tweaking
                    enc.base_scale.data.clamp_(min=0.001, max=0.015)
            
            
            metrics['g_loss'].append(loss_G.item())
            metrics['psnr'].append(calculate_psnr(cover, stego).item())
            
            preds = (torch.sigmoid(recovered_logits) > 0.5).float()
            valid_mask_bool = (scrambled_secret != -1.0) 
            ber = torch.abs(preds - scrambled_secret)[valid_mask_bool].sum() / (valid_mask_bool.sum() + 1e-8) * 100
            metrics['ber'].append(ber.item())
            
            pbar.set_description(f"E{epoch}")
            pbar.set_postfix(PSNR=f"{np.mean(metrics['psnr']):.1f}", BER=f"{np.mean(metrics['ber']):.2f}%")

        if epoch % 2 == 0 or epoch == c.epochs:
            visualize_result(enc, dec, full_ds, epoch)
            
        del stego, noisy_stego, recovered_logits
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(enc.state_dict(), 'stego_enc_gan.pth')
    torch.save(dec.state_dict(), 'stego_dec_gan.pth')
    print("✅ Training Complete. Models Saved.")

def visualize_result(enc, dec, ds, epoch):
    enc.eval(); dec.eval()
    idx = np.random.randint(0, len(ds))
    cover = ds[idx].unsqueeze(0).to(c.device)
    
    secret_text = '''
    {
      "Metadata": {
        "RecordID": "PRISM-HYBRID-487A",
        "Timestamp": "2025-11-28T12:48:09Z",
        "SourceSystem": "EHR-AlphaV2"
      },
      "PatientDemographics": {
        "ID": "PRISM-HYBRID",
        "Name": "Alex Johnson",
        "DateOfBirth": "1985-05-15",
        "Gender": "Male",
        "BloodType": "O+",
        "Contact": {
          "Phone": "555-0101",
          "Email": "alex.j@dummymail.com"
        }
      },
      "CurrentStatus": {
        "Status": "Critical",
        "AdmissionDate": "2025-11-25",
        "Ward": "ICU-3",
        "Diagnosis": "Severe Septic Shock secondary to unknown infection source."
      },
      "MedicalHistory": {
        "Allergies": [
          "Penicillin",
          "Latex"
        ],
        "Conditions": [
          "Type 2 Diabetes (controlled)",
          "Hypertension"
        ],
        "Surgeries": [
          {
            "Procedure": "Appendectomy",
            "Year": 2005
          },
          {
            "Procedure": "Left Knee Arthroscopy",
            "Year": 2018
          }
        ],
        "Medications": [
          {
            "Name": "Metformin (500mg)",
            "Dosage": "Twice Daily",
            "Start": "2020-01-10"
          },
          {
            "Name": "Lisinopril (10mg)",
            "Dosage": "Daily",
            "Start": "2021-03-01"
          }
        ]
      },
      "LatestVitals": {
        "Time": "2025-11-28T12:30:00Z",
        "Temperature_C": 38.9,
        "HeartRate_bpm": 115,
        "BloodPressure_mmHg": "85/50",
        "RespiratoryRate_rpm": 24,
        "OxygenSaturation_percent": 92
      },
      "LabResults": {
        "CompleteBloodCount": {
          "Time": "2025-11-28T08:00:00Z",
          "WBC": {
            "Value": 25.4,
            "Unit": "K/uL",
            "Range": "4.5-11.0"
          },
          "Hemoglobin": {
            "Value": 10.1,
            "Unit": "g/dL",
            "Range": "13.5-17.5"
          },
          "Platelets": {
            "Value": 98,
            "Unit": "K/uL",
            "Range": "150-450"
          }
        },
        "MetabolicPanel": {
          "Time": "2025-11-28T08:00:00Z",
          "Creatinine": {
            "Value": 1.9,
            "Unit": "mg/dL",
            "Range": "0.6-1.3"
          },
          "Glucose": {
            "Value": 310,
            "Unit": "mg/dL",
            "Range": "70-100"
          },
          "Lactate": {
            "Value": 4.8,
            "Unit": "mmol/L",
            "Range": "0.5-2.2"
          }
        }
      },
      "DoctorComments": [
        {
          "Physician": "Dr. Sarah Chen, MD",
          "Time": "2025-11-27T21:00:00Z",
          "Note": "Patient remains hemodynamically unstable despite max-dose vasopressors. Initial cultures are pending. Started empiric broad-spectrum antibiotics (Vancomycin + Meropenem). Hypoxia worsening, requires high-flow nasal cannula. Will repeat labs in 6 hours. Discussed prognosis with family, which is guarded."
        },
        {
          "Physician": "Dr. Mark Lee, DO (Consult)",
          "Time": "2025-11-28T10:30:00Z",
          "Note": "ID consult. Given the severity of shock and high lactate, infection source remains elusive. Consider imaging (CT chest/abdomen/pelvis) if patient can tolerate transport. If cultures remain negative and condition deteriorates, consider adding antifungal coverage. **Critical priority: volume resuscitation.**"
        }
      ]
    }
    '''
    
    secret_bits = PayloadEngine.str_to_bits(secret_text)
    capacity = c.cropsize * c.cropsize
    bits_list = [int(b) for b in secret_bits]
    header = format(len(bits_list), f'0{32}b')
    header_list = [int(b) for b in header] * 15 
    
    full_payload = header_list + bits_list
    actual_bpp = len(full_payload) / capacity
    
    if len(full_payload) < capacity:
        padding = [-1.0 for _ in range(capacity - len(full_payload))]
        full_payload.extend(padding)
    
    secret_tensor = torch.tensor(full_payload).float().view(1, 1, c.cropsize, c.cropsize).to(c.device)

    pool_idx = 0
    scrambled_tensor, _ = PayloadEngine.fast_scramble(secret_tensor, pool_idx=pool_idx)

    with torch.no_grad():
        stego, attn_map = enc(cover, scrambled_tensor)
        
        stego_quantized = torch.round(stego * 255.0) / 255.0
        rec_logits = dec(stego_quantized)
        
        rec_probs = torch.sigmoid(rec_logits)
        descrambled_tensor = PayloadEngine.fast_descramble(rec_probs, pool_idx=pool_idx)
        rec_text = PayloadEngine.decode(descrambled_tensor)
        
    psnr = calculate_psnr(cover, stego)
    
    valid_mask = (secret_tensor != -1.0)
    ber = torch.abs((descrambled_tensor > 0.5).float() - secret_tensor)[valid_mask].sum() / valid_mask.sum() * 100

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    residual = (stego - cover).abs() * 50 
    
    axs[0].imshow(cover.squeeze().permute(1,2,0).cpu().clamp(0,1))
    axs[0].set_title("Original Cover")
    
    axs[1].imshow(stego.squeeze().permute(1,2,0).cpu().clamp(0,1))
    axs[1].set_title(f"Stego (PSNR: {psnr:.2f}dB)")
    
    axs[2].imshow(residual.squeeze().permute(1,2,0).cpu().clamp(0,1))
    axs[2].set_title("Diff x50 (Tint Check)")
    
    plt.suptitle(f"Epoch {epoch} | BPP: {actual_bpp:.3f} | BER: {ber:.2f}%\nSent: {secret_text[:30]}...\nRec: {rec_text[:30]}...")
    plt.tight_layout()
    plt.show()
    plt.close(fig)      
    plt.clf()           
    gc.collect()        
    torch.cuda.empty_cache()

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    train_gan()