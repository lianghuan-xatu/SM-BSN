import torch
import cv2
from model.get_model import BSN
from analysis.scorecam.scorecam import ScoreCAM
from analysis.scorecam.utils import *

def load_model(model_path):
    # 加载MMBSN模型
    model = BSN(type='MMBSN')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_weight']['denoiser'], strict=False)
    model.eval()
    return model

def analyze_attention_distribution(model, input_image, target_layer='bsn.branch1_1.head.0'):
    # 准备模型字典，修改目标层的格式
    model_dict = dict(
        type='custom',
        arch=model,
        target_layer=target_layer.replace('.', '_'),  # 将点号替换为下划线
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

    # 添加CUDA内存清理
    torch.cuda.empty_cache()
    
    # 限制处理的特征图数量
    max_features = 100
    
    # 生成CAM图
    try:
        with torch.cuda.amp.autocast():  # 使用混合精度计算
            cam_map = score_cam(input_tensor)
            print(f"CAM图形状: {cam_map.shape}")
    except RuntimeError as e:
        print(f"生成CAM图时出错: {e}")
        torch.cuda.empty_cache()
        return None, None
    finally:
        # 确保清理GPU内存
        torch.cuda.empty_cache()
    
    # 转移数据到CPU并释放GPU内存
    cam_map = cam_map.cpu()
    input_tensor = input_tensor.cpu()
    model.cpu()
    
    # 保存可视化结果
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_map, cmap='viridis')
    plt.colorbar()
    plt.savefig('/opt/feature_map.png')
    plt.close()
    
    basic_visualize(input_tensor, cam_map.type(torch.FloatTensor), save_path='/opt/smbsn.png')
    
    # 最后再次清理内存
    torch.cuda.empty_cache()
    return None, None

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
        # 每次分析前清理GPU内存
        torch.cuda.empty_cache()
        print(f"\n分析层: {layer}")
        result, cam_map = analyze_attention_distribution(model, test_image, layer)
        # 每次分析后清理GPU内存
        torch.cuda.empty_cache()
        if result is not None:
            # 保存结果
            save_name = f'analysis/scorecam/results/{layer.replace(".", "_")}_analysis.png'
            cv2.imwrite(save_name, result)
            print(f'已保存{layer}层的分析结果到: {save_name}')