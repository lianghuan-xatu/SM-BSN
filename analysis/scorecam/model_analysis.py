import torch
import torch.nn.functional as F
import cv2
import numpy as np
from model.get_model import BSN
from cam.scorecam import ScoreCAM
from util.generator import np2tensor, tensor2np
from score_cam.utils import *

def load_model(model_path):
    # 加载MMBSN模型
    model = BSN(type='MMBSN')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_weight']['denoiser'], strict=False)
    model.eval()
    return model

def analyze_receptive_field(model, input_image, target_layer='bsn.branch1_1.head.0'):
    # 获取目标层
    target_module = model

    # 准备模型字典
    model_dict = dict(
        type='custom',
        arch=model,
        target_layer=target_layer,
        input_size=(225, 225)
    )

    score_cam = ScoreCAM(model_dict)

    # 修改输入预处理
    input_image = load_image(test_image)
    input_tensor = apply_transforms(input_image, 225)
    
    # 确保输入需要梯度计算
    input_tensor.requires_grad_(True)
    
    # 添加批次维度并转换为浮点类型
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.float()

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model = model.cuda()

    # 生成CAM图
    cam_map = score_cam(input_tensor)
    basic_visualize(input_tensor.cpu(), cam_map.type(torch.FloatTensor).cpu(), save_path='smbsn.png')

    return None, None

def basic_visualize(input_, gradients, save_path=None, cmap='viridis', alpha=0.7):
    # 确保数据在正确范围内
    input_ = torch.clamp(input_, 0, 1)  # 限制输入范围在 [0,1]
    gradients = torch.clamp(gradients, 0, 1)  # 限制梯度范围在 [0,1]
    
    # 转换为 numpy 数组
    input_ = input_.numpy()
    gradients = gradients.numpy()
    
    # 创建图像
    plt.figure(figsize=(10, 5))
    
    # 显示原始图像
    plt.subplot(121)
    plt.imshow(input_)
    plt.axis('off')
    plt.title('Original Image')
    
    # 显示热力图
    plt.subplot(122)
    plt.imshow(input_)
    plt.imshow(gradients, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.title('Attention Map')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # 示例使用
    model_path = '../../ckpt/MMBSN_SIDD_o_a45.pth'
    # test_image = '../../dataset/test_data/0001_18.png'
    test_image = '../../images/' + 'ILSVRC2012_val_00002193.JPEG'
    
    # 加载模型
    model = load_model(model_path)
    
    # 分析不同层的感受野
    layers_to_analyze = [
        'branch1_1',  # 第一个去噪卷积层
    ]
    
    for layer in layers_to_analyze:
        result, cam_map = analyze_receptive_field(model, test_image, layer)
        if result is not None:
            # 保存结果
            save_name = f'analysis/scorecam/results/{layer.replace(".", "_")}_analysis.png'
            cv2.imwrite(save_name, result)
            print(f'已保存{layer}层的分析结果到: {save_name}')