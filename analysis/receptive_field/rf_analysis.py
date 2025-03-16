# A script to visualize the ERF (Effective Receptive Field).
# 参考自: Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# 改编自: https://github.com/DingXiaoH/RepLKNet-pytorch
# --------------------------------------------------------'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from model.get_model import BSN
from analysis.scorecam.utils import load_image, apply_transforms
from torchvision import transforms
from PIL import Image
from timm.utils import AverageMeter
import seaborn as sns

# 设置图表参数
# 修改字体设置
plt.rcParams["font.family"] = "DejaVu Sans"  # 使用更通用的字体
large = 24; med = 24; small = 24
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,
          'axes.grid': True,
          'grid.color': '#E5E5E5',
          'axes.facecolor': 'white',
          'axes.edgecolor': '#737373',
          'axes.linewidth': 1.0,
          'grid.alpha': 0.5}
plt.rcParams.update(params)
sns.set_theme(style="white")
plt.rcParams['axes.unicode_minus'] = False

def load_model(model_path):
    """加载MMBSN模型"""
    model = BSN(type='MMBSN')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_weight']['denoiser'], strict=False)
    model.eval()
    return model

def get_input_grad(model, samples, layer_name=None):
    """计算输入梯度，用于ERF可视化"""
    # 如果指定了特定层，则获取该层的输出
    if layer_name:
        # 查找目标层
        target_layer = None
        for name, module in model.named_modules():
            if layer_name in name:
                target_layer = module
                print(f"找到目标层: {name}")
                break
        
        if target_layer is None:
            raise ValueError(f"找不到层 {layer_name}")
        
        # 注册钩子以获取特征图
        activations = []
        def forward_hook(module, input, output):
            activations.append(output)
        
        handle = target_layer.register_forward_hook(forward_hook)
        
        # 前向传播
        _ = model(samples)
        
        # 获取特征图
        if not activations:
            handle.remove()
            raise ValueError("没有获取到特征图")
        
        feature_map = activations[0]
        
        # 选择中心点的特征
        out_size = feature_map.size()
        central_point = torch.nn.functional.relu(
            feature_map[:, :, out_size[2] // 2, out_size[3] // 2]
        ).sum()
        
        # 清除钩子
        handle.remove()
    else:
        # 如果没有指定层，则使用模型的最终输出
        outputs = model(samples)
        out_size = outputs.size()
        central_point = torch.nn.functional.relu(
            outputs[:, :, out_size[2] // 2, out_size[3] // 2]
        ).sum()
    
    # 修改梯度处理方式
    grad = torch.autograd.grad(central_point, samples)[0]
    
    # 使用绝对值而不是ReLU，保留负梯度的信息
    grad = grad.abs()
    # 对每个通道分别归一化后再求和
    grad = grad / (grad.sum((2, 3), keepdim=True) + 1e-8)
    aggregated = grad.sum(1).mean(0)  # 先对通道求和，再对batch求平均
    grad_map = aggregated.cpu().numpy()
    
    return grad_map

def compute_erf(model, layer_name=None, num_images=10, input_size=(225, 225), test_image=None):
    """计算有效感受野(ERF)"""
    # 创建平均计量器
    meter = AverageMeter()
    
    # 加载测试图像
    test_images = []
    if test_image and os.path.isdir(test_image):
        # 如果是目录，加载多张图像
        from glob import glob
        image_files = glob(os.path.join(test_image, '*.jpg')) + glob(os.path.join(test_image, '*.png'))
        for img_path in image_files[:num_images]:
            img = load_image(img_path)
            tensor = apply_transforms(img, input_size[0])
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            test_images.append(tensor)
    elif test_image:
        # 单张图像，创建多个变换版本
        img = load_image(test_image)
        for _ in range(num_images):
            # 应用随机变换以获得不同版本
            transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            tensor = transform(img).unsqueeze(0)
            test_images.append(tensor)
    else:
        # 如果没有提供图像，创建随机噪声图像
        for _ in range(num_images):
            tensor = torch.randn(1, 3, input_size[0], input_size[1])
            tensor = tensor / tensor.std()  # 标准化
            test_images.append(tensor)
    
    # 处理每张图像
    for i, img_tensor in enumerate(test_images):
        print(f"处理图像 {i+1}/{len(test_images)}")
        
        # 确保需要梯度
        img_tensor.requires_grad_(True)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            model = model.cuda()
        
        # 清除梯度
        model.zero_grad()
        
        # 计算输入梯度
        contribution_scores = get_input_grad(model, img_tensor, layer_name)
        
        # 检查NaN
        if np.isnan(np.sum(contribution_scores)):
            print('发现NaN值，跳过此图像')
            continue
        
        # 累积贡献分数
        meter.update(contribution_scores)
        
        # 清理内存
        torch.cuda.empty_cache()
    
    return meter.avg

def heatmap(data, camp='RdBu_r', figsize=(10, 10.75), ax=None, save_path=None):
    """绘制热力图"""
    plt.figure(figsize=figsize, dpi=100)
    
    # 对数据进行预处理
    data = data - np.min(data)  # 确保数据非负
    data = data / (np.max(data) + 1e-8)  # 归一化到 [0,1]
    
    # 创建对称的色彩范围
    vmax = 1.0
    vmin = 0.0
    
    ax = sns.heatmap(data,
                     xticklabels=False,
                     yticklabels=False, 
                     cmap='viridis',  # 使用viridis色图效果更好
                     vmin=vmin,
                     vmax=vmax,
                     annot=False,
                     ax=ax,
                     cbar_kws={'label': 'Contribution'})
    
    plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', pad=0.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 提高DPI
    plt.close()

def get_rectangle(data, threshold):
    """计算高贡献区域的矩形范围
    
    Args:
        data: 贡献度数据
        threshold: 贡献度阈值
    
    Returns:
        tuple: (边长, 面积比例) 如果找到合适的区域，否则返回None
    """
    # 获取高于阈值的区域
    high_contribution = data > threshold
    
    if not high_contribution.any():
        return None
    
    # 找到高贡献区域的边界
    rows = np.any(high_contribution, axis=1)
    cols = np.any(high_contribution, axis=0)
    
    # 获取边界索引
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 计算矩形边长（取较大值）
    side_length = max(rmax - rmin + 1, cmax - cmin + 1)
    
    # 计算面积比例
    total_area = data.shape[0] * data.shape[1]
    high_area = np.sum(high_contribution)
    area_ratio = high_area / total_area
    
    return side_length, area_ratio

def analyze_erf(erf_data, save_path):
    """分析ERF数据并生成可视化"""
    from scipy.ndimage import gaussian_filter
    
    # 数据预处理
    erf_data = np.abs(erf_data)  # 取绝对值
    
    # 增强中心化处理
    h, w = erf_data.shape
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    # 增大高斯mask的标准差，使感受野范围更大
    sigma = min(h,w)/3  # 增大标准差
    mask = np.exp(-(x*x + y*y) / (2*sigma**2))
    
    # 应用更强的高斯平滑
    smoothed_data = gaussian_filter(erf_data, sigma=5)  # 增大sigma值
    
    # 应用中心权重
    processed_data = smoothed_data * mask
    
    # 改进的归一化方法
    processed_data = processed_data - np.min(processed_data)
    processed_data = processed_data / (np.max(processed_data) + 1e-8)
    
    print('======================= 高贡献区域比例 =====================')
    for thresh in [0.2, 0.3, 0.5, 0.99]:
        result = get_rectangle(processed_data, thresh)
        if result:
            side_length, area_ratio = result
            print(f'阈值: {thresh}, 矩形边长: {side_length}, 面积比例: {area_ratio:.4f}')
    
    # 保存热力图，使用不同的颜色映射
    heatmap(processed_data, camp='RdBu_r', save_path=save_path)
    print(f'热力图已保存至 {save_path}')
    
    return processed_data

def visualize_receptive_field(model, input_image, layer_name, num_images=5, save_dir='/opt/receptive_fields'):
    """可视化感受野"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成安全的文件名
    safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
    
    # 计算ERF
    print(f"计算层 {layer_name} 的有效感受野...")
    erf_data = compute_erf(model, layer_name, num_images=num_images, test_image=input_image)
    
    # 保存原始ERF数据
    npy_path = os.path.join(save_dir, f"{safe_layer_name}_erf.npy")
    np.save(npy_path, erf_data)
    print(f"ERF数据已保存到: {npy_path}")
    
    # 分析并可视化ERF
    heatmap_path = os.path.join(save_dir, f"{safe_layer_name}_erf_heatmap.png")
    processed_data = analyze_erf(erf_data, heatmap_path)
    
    # 创建3D可视化
    plt.figure(figsize=(16, 10), dpi=40)
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')
    
    # 准备数据
    h, w = processed_data.shape
    x = np.arange(0, w)
    y = np.arange(0, h)
    x, y = np.meshgrid(x, y)
    
    # 绘制3D表面
    surf = ax.plot_surface(x, y, processed_data, cmap='viridis', linewidth=0, antialiased=False)
    
    # 添加颜色条
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # 设置标题和标签
    plt.title(f'Layer: {layer_name} - 3D ERF Visualization')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Contribution')
    
    # 保存3D图
    surface_path = os.path.join(save_dir, f"{safe_layer_name}_erf_3d.png")
    plt.savefig(surface_path, dpi=300)
    plt.close()
    print(f"3D可视化已保存到: {surface_path}")
    
    return erf_data

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('用于可视化有效感受野(ERF)的脚本')
    parser.add_argument('--model_path', default='../../ckpt/MMBSN_SIDD_o_a45.pth', type=str, help='模型权重路径')
    # 修改默认的测试图像路径
    # parser.add_argument('--test_image',
    #                    default='./analysis/receptive_field/images/9_7_4.png',
    #                    type=str, help='测试图像路径')
    parser.add_argument('--test_image',
                       default='/opt/modules/dataset/prep/DND_s512_o128/RN/',
                       type=str, help='测试图像路径')
    # 修改默认的保存路径
    parser.add_argument('--save_dir', 
                       default='/opt/modules/receptive_fields',
                       type=str, help='保存结果的目录')
    parser.add_argument('--num_images', default=30, type=int, help='用于计算ERF的图像数量')
    args = parser.parse_args()
    
    # 确保测试图像路径是绝对路径
    if not os.path.isabs(args.test_image):
        args.test_image = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     args.test_image)
    
    return args

if __name__ == '__main__':
    # 解析参数
    args = parse_args()
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'shallow'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'deep'), exist_ok=True)
    
    # 加载模型
    model = load_model(args.model_path)
    
    # 自动获取所有浅层
    shallow_layers = []
    for name, module in model.named_modules():
        # 选择浅层卷积层
        if isinstance(module, nn.Conv2d) and ('branch1' in name or 'branch2' in name):
            if 'head' not in name and 'tail' not in name:  # 排除头尾层
                shallow_layers.append(name)
    
    # 自动获取中间层（移到这里）
    middle_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 主干网络的中间层
            if 'body.1' in name or 'body.2' in name:
                middle_layers.append(name)
            # 分支网络的中间层
            elif 'branch' in name and ('conv2' in name or 'conv3' in name):
                middle_layers.append(name)
            # 融合层
            elif 'fusion' in name and not ('tail' in name or 'head' in name):
                middle_layers.append(name)
    
    # 自动获取更深层
    deep_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'tail' in name or 'fusion' in name:
                deep_layers.append(name)
            elif ('branch' in name and 'body.2' in name) or ('branch' in name and 'conv1_3' in name):
                deep_layers.append(name)
            elif 'output' in name or 'combine' in name or 'merge' in name:
                deep_layers.append(name)
            elif 'body.3' in name or 'body.4' in name:
                deep_layers.append(name)
    
    # 获取特殊层（如果有的话）
    special_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'combine' in name:
            special_layers.append(name)
    
    print(f"找到 {len(shallow_layers)} 个浅层卷积层")
    print(f"找到 {len(deep_layers)} 个深层卷积层")
    if special_layers:
        print(f"找到 {len(special_layers)} 个特殊层")
    
    # 创建结果目录
    import os
    save_dir = '/opt/receptive_fields'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'shallow'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'deep'), exist_ok=True)
    if special_layers:
        os.makedirs(os.path.join(save_dir, 'special'), exist_ok=True)
    
    # 分析所有层
    results = {}
    saved_images = []
    
    # 首先分析浅层
    print("\n===== 分析浅层 =====")
    for layer in shallow_layers:
        print(f"\n分析层 {layer} 的感受野")
        try:
            erf_data = visualize_receptive_field(
                model, 
                args.test_image,  # 使用args中的test_image
                layer, 
                num_images=args.num_images,
                save_dir=os.path.join(args.save_dir, 'shallow')
            )
            results[layer] = erf_data
            
            # 检查图像是否已保存
            safe_layer_name = layer.replace('.', '_').replace('/', '_')
            image_path = os.path.join(args.save_dir, 'shallow', f"{safe_layer_name}_erf_heatmap.png")
            if os.path.exists(image_path):
                saved_images.append(image_path)
                print(f"ERF分析完成，热力图已保存")
            else:
                print(f"ERF分析完成，但热力图保存失败")
        except Exception as e:
            print(f"分析层 {layer} 时出错: {e}")
    
    # 分析中间层
    print("\n===== 分析中间层 =====")
    for layer in middle_layers:
        print(f"\n分析层 {layer} 的感受野")
        try:
            erf_data = visualize_receptive_field(
                model,
                args.test_image,  # 使用args中的test_image
                layer,
                num_images=args.num_images,
                save_dir=os.path.join(args.save_dir, 'middle')
            )
            results[layer] = erf_data
            
            # 检查图像是否已保存
            safe_layer_name = layer.replace('.', '_').replace('/', '_')
            image_path = os.path.join(args.save_dir, 'middle', f"{safe_layer_name}_erf_heatmap.png")
            if os.path.exists(image_path):
                saved_images.append(image_path)
                print(f"ERF分析完成，热力图已保存")
            else:
                print(f"ERF分析完成，但热力图保存失败")
        except Exception as e:
            print(f"分析层 {layer} 时出错: {e}")

    # 修改分析深层的部分
    print("\n===== 分析深层 =====")
    for layer in deep_layers:
        print(f"\n分析层 {layer} 的感受野")
        try:
            erf_data = visualize_receptive_field(
                model,
                args.test_image,  # 使用args中的test_image
                layer,
                num_images=args.num_images,
                save_dir=os.path.join(args.save_dir, 'deep')
            )
            results[layer] = erf_data
            
            # 检查图像是否已保存
            safe_layer_name = layer.replace('.', '_').replace('/', '_')
            image_path = os.path.join(args.save_dir, 'deep', f"{safe_layer_name}_erf_heatmap.png")
            if os.path.exists(image_path):
                saved_images.append(image_path)
                print(f"ERF分析完成，热力图已保存")
            else:
                print(f"ERF分析完成，但热力图保存失败")
        except Exception as e:
            print(f"分析层 {layer} 时出错: {e}")

    # 创建中间层的保存目录
    os.makedirs(os.path.join(save_dir, 'middle'), exist_ok=True)
    
    # 分析中间层
    print("\n===== 分析中间层 =====")
    for layer in middle_layers:
        print(f"\n分析层 {layer} 的感受野")
        try:
            rf_h, rf_w = visualize_receptive_field(model, test_image, layer, os.path.join(save_dir, 'middle'))
            results[layer] = (rf_h, rf_w, "中间层")
            
            # 检查图像是否已保存
            safe_layer_name = layer.replace('.', '_').replace('/', '_')
            image_path = os.path.join(save_dir, 'middle', f"{safe_layer_name}_rf.png")
            if os.path.exists(image_path):
                saved_images.append(image_path)
                print(f"感受野大小: {rf_h}x{rf_w}, 图像已保存")
            else:
                print(f"感受野大小: {rf_h}x{rf_w}, 但图像保存失败")
        except Exception as e:
            print(f"分析层 {layer} 时出错: {e}")
    
    # 最后分析特殊层
    if special_layers:
        print("\n===== 分析特殊层 =====")
        for layer in special_layers:
            print(f"\n分析层 {layer} 的感受野")
            try:
                erf_data = visualize_receptive_field(
                    model,
                    args.test_image,
                    layer,
                    num_images=args.num_images,
                    save_dir=os.path.join(args.save_dir, 'special')
                )
                results[layer] = erf_data
                
                # 检查图像是否已保存
                safe_layer_name = layer.replace('.', '_').replace('/', '_')
                image_path = os.path.join(args.save_dir, 'special', f"{safe_layer_name}_erf_heatmap.png")
                if os.path.exists(image_path):
                    saved_images.append(image_path)
                    print(f"ERF分析完成，热力图已保存")
                else:
                    print(f"ERF分析完成，但热力图保存失败")
            except Exception as e:
                print(f"分析层 {layer} 时出错: {e}")
    
    # 添加最深层分析
    print("\n===== 分析最深层 =====")
    deepest_layers = []
    final_output_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 选择最终输出层
            if any(x in name.lower() for x in ['output', 'final', 'predict', 'out']):
                final_output_layers.append(name)
            # 选择最深的卷积层
            elif 'tail' in name or 'fusion.tail' in name:
                deepest_layers.append(name)
            # 添加更多深层
            elif any(x in name for x in ['combine.tail', 'merge.tail', 'fusion.out']):
                deepest_layers.append(name)
            # 添加BSN特有的最终层
            elif 'bsn.final' in name or 'bsn.output' in name:
                final_output_layers.append(name)
    
    print(f"找到 {len(deepest_layers)} 个最深层")
    print(f"找到 {len(final_output_layers)} 个最终输出层")
    os.makedirs(os.path.join(args.save_dir, 'deepest'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'final_output'), exist_ok=True)
    
    # 先分析最深层
    for layer in deepest_layers:
        print(f"\n分析层 {layer} 的感受野")
        try:
            erf_data = visualize_receptive_field(
                model,
                args.test_image,
                layer,
                num_images=args.num_images,
                save_dir=os.path.join(args.save_dir, 'deepest')
            )
            results[layer] = erf_data
            
            safe_layer_name = layer.replace('.', '_').replace('/', '_')
            image_path = os.path.join(args.save_dir, 'deepest', f"{safe_layer_name}_erf_heatmap.png")
            if os.path.exists(image_path):
                saved_images.append(image_path)
                print(f"ERF分析完成，热力图已保存")
            else:
                print(f"ERF分析完成，但热力图保存失败")
        except Exception as e:
            print(f"分析层 {layer} 时出错: {e}")
    
    # 分析最终输出层
    print("\n===== 分析最终输出层 =====")
    for layer in final_output_layers:
        print(f"\n分析层 {layer} 的感受野")
        try:
            erf_data = visualize_receptive_field(
                model,
                args.test_image,
                layer,
                num_images=args.num_images,
                save_dir=os.path.join(args.save_dir, 'final_output')
            )
            results[layer] = erf_data
            
            safe_layer_name = layer.replace('.', '_').replace('/', '_')
            image_path = os.path.join(args.save_dir, 'final_output', f"{safe_layer_name}_erf_heatmap.png")
            if os.path.exists(image_path):
                saved_images.append(image_path)
                print(f"ERF分析完成，热力图已保存")
            else:
                print(f"ERF分析完成，但热力图保存失败")
        except Exception as e:
            print(f"分析层 {layer} 时出错: {e}")

    print(f"\n成功保存了 {len(saved_images)} 个感受野分析图")
    for img in saved_images[:5]:  # 只显示前5个
        print(f" - {img}")
    if len(saved_images) > 5:
        print(f" - ... 以及其他 {len(saved_images)-5} 个图像")
    
    # 修改保存 CSV 的部分
    import csv
    with open(os.path.join(save_dir, 'receptive_field_results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'ERF Shape', 'Max Value', 'Min Value'])
        for layer, erf_data in results.items():
            writer.writerow([
                layer, 
                str(erf_data.shape),
                f"{np.max(erf_data):.4f}",
                f"{np.min(erf_data):.4f}"
            ])
    
    print(f"\n所有感受野分析结果已保存到 {save_dir}")