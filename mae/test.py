import sys
import os

import openslide
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
	# image is [H, W, 3]
	assert image.shape[2] == 3
	plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
	plt.title(title, fontsize=16)
	plt.axis('off')
	return


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
	# build model
	model = getattr(models_mae, arch)()
	# load model
	checkpoint = torch.load(chkpt_dir, map_location='cpu')
	msg = model.load_state_dict(checkpoint['model'], strict=False)
	print(msg)
	return model


def run_one_image(img, model):
	x = torch.tensor(img)
	
	# make it a batch-like
	x = x.unsqueeze(dim=0)
	x = torch.einsum('nhwc->nchw', x)
	
	# run MAE
	loss, y, mask = model(x.float(), mask_ratio=0.75)
	y = model.unpatchify(y)
	y = torch.einsum('nchw->nhwc', y).detach().cpu()
	
	# visualize the mask
	mask = mask.detach()
	mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
	mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
	mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
	
	x = torch.einsum('nchw->nhwc', x)
	
	# masked image
	im_masked = x * (1 - mask)
	
	# MAE reconstruction pasted with visible patches
	im_paste = x * (1 - mask) + y * mask
	
	# make the plt figure larger
	plt.rcParams['figure.figsize'] = [24, 24]
	
	plt.subplot(1, 4, 1)
	show_image(x[0], "original")
	
	plt.subplot(1, 4, 2)
	show_image(im_masked[0], "masked")
	
	plt.subplot(1, 4, 3)
	show_image(y[0], "reconstruction")
	
	plt.subplot(1, 4, 4)
	show_image(im_paste[0], "reconstruction + visible")
	
	plt.show()

if __name__ == '__main__':
	model = prepare_model('checkpoint-100.pth')
	print(model)
	
	path = r'E:\project\bioProject_breast\slide\TCGA-E2-A572-01A-01-TSA.58C737CB-AF59-4664-9E55-2097E80A35FA.svs'
	slide = openslide.open_slide(path)
	
	print(slide)
	sslide = slide.get_thumbnail((2067,440))
	plt.imshow(sslide)
	plt.show()
	
	img = slide.read_region((19600, 12100), 0, (256, 256)).convert('RGB')
	plt.imshow(img)
	plt.show()

	img = img.resize((224, 224))
	img = np.array(img) / 255.
	img = img - imagenet_mean
	img = img / imagenet_std
	#show_image(torch.tensor(img))
	run_one_image(img,model)