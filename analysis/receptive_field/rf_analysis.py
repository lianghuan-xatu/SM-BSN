import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model.get_model import BSN
from analysis.scorecam.utils import load_image, apply_transforms

def load_model(model_path):
    model = BSN(type='MMBSN')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_weight']['denoiser'], strict=False)
    model.eval()
    return model

def compute_receptive_field(model, layer_name, input_size=(225, 225)):
    def hook_fn(module, input, output):
        hook_fn.feature_maps = output.detach()
    
    # 打印所有可用的层名称，帮助调试
    print("模型中的所有层:")
    available_layers = []
    for name, module in model.named_modules():
        available_layers.append(name)
        if len(available_layers) <= 10:  # 只打印前10个作为示例
            print(f"  - {name}: {type(module).__name__}")
    print(f"总共找到 {len(available_layers)} 个层")
    
    # 尝试不同的层名称格式
    target_layer = None
    possible_names = [
        layer_name,
        layer_name.replace('bsn.', ''),  # 尝试移除前缀
        'bsn.' + layer_name if not layer_name.startswith('bsn.') else layer_name,  # 尝试添加前缀
        layer_name.replace('.', '_')  # 尝试使用下划线替代点号
    ]
    
    for name in possible_names:
        for module_name, module in model.named_modules():
            if name == module_name:
                target_layer = module
                print(f"找到目标层: {module_name}")
                break
        if target_layer is not None:
            break
    
    if target_layer is None:
        print(f"警告: 找不到层 '{layer_name}'，尝试查找包含该名称的层")
        # 尝试查找包含该名称的层
        for module_name, module in model.named_modules():
            if layer_name.split('.')[-1] in module_name:
                target_layer = module
                print(f"找到相似层: {module_name}")
                break
    
    if target_layer is None:
        raise ValueError(f"层 {layer_name} 未找到。可用的层: {available_layers[:10]}...")
    
    # 注册钩子
    handle = target_layer.register_forward_hook(hook_fn)
    
    # 创建高斯脉冲输入而不是方形脉冲
    input_tensor = torch.zeros((1, 3, input_size[0], input_size[1]))
    center_h, center_w = input_size[0] // 2, input_size[1] // 2
    
    # 使用高斯分布的脉冲
    sigma = 2.0  # 高斯分布的标准差
    radius = 10  # 影响范围半径
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if 0 <= center_h + i < input_size[0] and 0 <= center_w + j < input_size[1]:
                # 创建高斯分布的脉冲
                g = np.exp(-(i**2 + j**2) / (2 * sigma**2))
                input_tensor[0, :, center_h + i, center_w + j] = g * 10.0
    
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model = model.cuda()
    
    # 前向传播
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 获取特征图
    feature_maps = hook_fn.feature_maps
    
    # 计算感受野大小
    # 计算感受野大小时使用更严格的阈值
    feature_map = feature_maps[0, 0].cpu().numpy()
    threshold = feature_map.mean() + 0.1 * feature_map.std()  # 使用均值+0.1倍标准差作为阈值
    active_pixels = np.where(feature_map > threshold)
    
    if len(active_pixels[0]) > 0:
        rf_height = active_pixels[0].max() - active_pixels[0].min()
        rf_width = active_pixels[1].max() - active_pixels[1].min()
    else:
        rf_height = rf_width = 0
    
    # 清除钩子
    handle.remove()
    
    return rf_height, rf_width, feature_map, threshold  # 返回阈值

def visualize_receptive_field(model, input_image, layer_name, save_dir='/opt/receptive_fields'):
    # 加载和预处理输入图像
    input_image_pil = load_image(input_image)
    input_tensor = apply_transforms(input_image_pil, 225)
    
    # 确保输入张量维度正确
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
    
    # 禁用 CUDNN 自动调优以避免警告
    torch.backends.cudnn.enabled = False
    
    # 计算感受野
    rf_height, rf_width, feature_map, threshold = compute_receptive_field(model, layer_name)
    
    # 创建保存目录
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成安全的文件名
    safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
    save_path = os.path.join(save_dir, f"{safe_layer_name}_rf.png")
    
    # 导入所需库
    from skimage.transform import resize
    from scipy import ndimage
    from matplotlib.colors import LogNorm
    
    # 将PIL图像转换为numpy数组以获取形状
    input_np = np.array(input_image_pil)
    
    # 感受野可视化增强
    binary_map = feature_map > threshold
    
    # 使用膨胀操作连接相邻的点
    dilated_map = ndimage.binary_dilation(binary_map, iterations=1)
    
    # 创建第一个图形
    plt.figure(figsize=(20, 5))
    
    # 原始图像
    plt.subplot(141)
    plt.imshow(input_image_pil)
    plt.title('Input Image')
    
    # 特征图可视化增强
    plt.subplot(142)
    # 确保LogNorm参数有效
    min_val = max(feature_map.min(), 1e-10)  # 避免负值或零
    max_val = max(feature_map.max(), min_val + 1e-5)  # 确保max > min
    plt.imshow(feature_map, cmap='viridis', norm=LogNorm(vmin=min_val, vmax=max_val))
    plt.colorbar()
    plt.title('Feature Map (Log Scale)')
    
    # 感受野可视化
    plt.subplot(143)
    plt.imshow(binary_map, cmap='gray')
    plt.title(f'Receptive Field\nHeight: {rf_height}, Width: {rf_width}\nActive Pixels: {np.sum(binary_map)}')
    
    # 感受野叠加到原图
    plt.subplot(144)
    overlay_map = resize(binary_map.astype(float), 
                         (input_np.shape[0], input_np.shape[1]), 
                         order=0, preserve_range=True)
    plt.imshow(input_image_pil)
    plt.imshow(overlay_map, cmap='hot', alpha=0.5)
    plt.title('Receptive Field Overlay')
    
    # 添加更多信息到图像
    plt.tight_layout()
    plt.suptitle(f'Layer: {layer_name}', fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    # 保存第一个图形
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # 创建第二个图形 - 膨胀分析
    dilated_save_path = os.path.join(save_dir, f"{safe_layer_name}_dilated_rf.png")
    
    try:
        plt.figure(figsize=(25, 5))
        
        # 原始图像
        plt.subplot(151)
        plt.imshow(input_image_pil)
        plt.title('Input Image')
        
        # 特征图
        plt.subplot(152)
        plt.imshow(feature_map, cmap='viridis', norm=LogNorm(vmin=min_val, vmax=max_val))
        plt.colorbar()
        plt.title('Feature Map (Log Scale)')
        
        # 原始感受野
        plt.subplot(153)
        plt.imshow(binary_map, cmap='gray')
        plt.title(f'Original Receptive Field\nActive Pixels: {np.sum(binary_map)}')
        
        # 膨胀感受野
        plt.subplot(154)
        plt.imshow(dilated_map, cmap='gray')
        plt.title(f'Dilated Receptive Field\nActive Pixels: {np.sum(dilated_map)}')
        
        # 膨胀感受野叠加
        plt.subplot(155)
        overlay_dilated = resize(dilated_map.astype(float), 
                                (input_np.shape[0], input_np.shape[1]), 
                                order=0, preserve_range=True)
        plt.imshow(input_image_pil)
        plt.imshow(overlay_dilated, cmap='hot', alpha=0.5)
        plt.title('Dilated RF Overlay')
        
        # 添加更多信息
        plt.tight_layout()
        plt.suptitle(f'Layer: {layer_name} (Dilated Analysis)', fontsize=16)
        plt.subplots_adjust(top=0.85)
        
        # 保存第二个图形
        plt.savefig(dilated_save_path, dpi=300)
        plt.close()
        
        print(f"已成功保存膨胀感受野分析图: {dilated_save_path}")
    except Exception as e:
        print(f"警告: 膨胀感受野分析图保存失败: {e}")
    
    # 检查文件是否成功保存
    if os.path.exists(save_path):
        print(f"已成功保存感受野分析图: {save_path}")
    else:
        print(f"警告: 感受野分析图保存失败: {save_path}")
    
    # 重新启用 CUDNN
    torch.backends.cudnn.enabled = True
    
    # 清理 GPU 内存
    torch.cuda.empty_cache()
    
    return rf_height, rf_width

if __name__ == '__main__':
    model_path = '../../ckpt/MMBSN_SIDD_o_a45.pth'
    test_image = '../../images/ILSVRC2012_val_00002193.JPEG'
    
    model = load_model(model_path)
    
    # 自动获取所有浅层
    shallow_layers = []
    for name, module in model.named_modules():
        # 选择浅层卷积层
        if isinstance(module, nn.Conv2d) and ('branch1' in name or 'branch2' in name):
            if 'head' not in name and 'tail' not in name:  # 排除头尾层
                shallow_layers.append(name)
    
    print(f"找到 {len(shallow_layers)} 个浅层卷积层")
    
    # 创建结果目录
    import os
    save_dir = '/opt/receptive_fields'
    os.makedirs(save_dir, exist_ok=True)
    
    # 分析所有浅层
    results = {}
    saved_images = []
    
    for layer in shallow_layers:
        print(f"\n分析层 {layer} 的感受野")
        try:
            rf_h, rf_w = visualize_receptive_field(model, test_image, layer, save_dir)
            results[layer] = (rf_h, rf_w)
            
            # 检查图像是否已保存
            safe_layer_name = layer.replace('.', '_').replace('/', '_')
            image_path = os.path.join(save_dir, f"{safe_layer_name}_rf.png")
            if os.path.exists(image_path):
                saved_images.append(image_path)
                print(f"感受野大小: {rf_h}x{rf_w}, 图像已保存")
            else:
                print(f"感受野大小: {rf_h}x{rf_w}, 但图像保存失败")
        except Exception as e:
            print(f"分析层 {layer} 时出错: {e}")
    
    print(f"\n成功保存了 {len(saved_images)} 个感受野分析图")
    for img in saved_images[:5]:  # 只显示前5个
        print(f" - {img}")
    if len(saved_images) > 5:
        print(f" - ... 以及其他 {len(saved_images)-5} 个图像")
    
    # 保存所有结果到CSV文件
    import csv
    with open(os.path.join(save_dir, 'receptive_field_results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'RF Height', 'RF Width'])
        for layer, (rf_h, rf_w) in results.items():
            writer.writerow([layer, rf_h, rf_w])
    
    print(f"\n所有感受野分析结果已保存到 {save_dir}")