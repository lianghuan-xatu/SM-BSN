import torch
import cv2
import numpy as np
from model.get_model import BSN
from analysis.gradcam.gradcam import GradCAM

import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(root_dir, 'ckpt', 'MMBSN_SIDD_o_a45.pth')
checkpoint = torch.load(model_path, map_location='cpu')
print(checkpoint.keys())  # 如果是字典的话，打印所有的键
# 查看模型权重
if 'model_weight' in checkpoint:
    state_dict = checkpoint['model_weight']
    print("\n模型权重的键值:", state_dict.keys())
    if 'denoiser' in state_dict:
        print(state_dict['denoiser'].keys())




