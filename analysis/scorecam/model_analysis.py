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

    # 添加调试信息
    with torch.no_grad():
        _ = model(input_tensor)
        activations = score_cam.activations['value']
        print(f"特征图形状: {activations.shape}")
        
        # 检查特征图的空间分布
        feature_map = activations[0, 0].cpu().numpy()
        print(f"特征图空间维度: {feature_map.shape}")
        print(f"特征图统计信息:")
        print(f"- 最大值: {feature_map.max():.4f}")
        print(f"- 最小值: {feature_map.min():.4f}")
        print(f"- 均值: {feature_map.mean():.4f}")
        print(f"- 标准差: {feature_map.std():.4f}")

    # 生成CAM图
    cam_map = score_cam(input_tensor)
    print(f"CAM图形状: {cam_map.shape}")
    
    # 保存原始特征图
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_map, cmap='viridis')
    plt.colorbar()
    plt.savefig('/opt/feature_map.png')
    plt.close()
    
    basic_visualize(input_tensor.cpu(), cam_map.type(torch.FloatTensor).cpu(), save_path='/opt/smbsn.png')

    return None, None

if __name__ == '__main__':
    # 示例使用
    model_path = '../../ckpt/MMBSN_SIDD_o_a45.pth'
    test_image = '../../images/' + 'ILSVRC2012_val_00002193.JPEG'
    
    # 加载模型
    model = load_model(model_path)
    
    # 分析不同层的感受野
    layers_to_analyze = [
        'branch1_1',  # 尝试不同的目标层
        # 'bsn.branch1_1.conv1',
        # 'bsn.branch1_1'
    ]
    
    for layer in layers_to_analyze:
        print(f"\n分析层: {layer}")
        result, cam_map = analyze_receptive_field(model, test_image, layer)
        if result is not None:
            # 保存结果
            save_name = f'analysis/scorecam/results/{layer.replace(".", "_")}_analysis.png'
            cv2.imwrite(save_name, result)
            print(f'已保存{layer}层的分析结果到: {save_name}')