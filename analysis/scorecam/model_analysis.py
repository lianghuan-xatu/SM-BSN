import torch
import torch.nn.functional as F
import cv2
import numpy as np
from model.get_model import BSN
from cam.scorecam import ScoreCAM
from util.generator import np2tensor, tensor2np

def load_model(model_path):
    # 加载MMBSN模型
    model = BSN(type='MMBSN')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_weight']['denoiser'], strict=False)
    model.eval()
    return model

def pad_to_multiple(x, factor=5):
    """将输入张量填充到能被factor整除的尺寸
    
    Args:
        x: 输入张量
        factor: 填充因子
    """
    h, w = x.shape[-2:]
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x

def analyze_receptive_field(model, input_image, target_layer='bsn.branch1_1.head.0'):
    """使用ScoreCAM分析MMBSN模型的感受野
    Args:
        model: MMBSN模型实例
        input_image: 输入图像路径
        target_layer: 要分析的目标层名称，例如'bsn.branch1_1.head.0'表示第一个分支的第一层
    """
    # 获取目标层
    target_module = model

    # 准备模型字典
    model_dict = dict(
        type='custom',
        arch=model,
        target_layer=target_layer,
        input_size=(224, 224)
    )

    score_cam = ScoreCAM(model_dict)
    
    # 加载并预处理图像
    if isinstance(input_image, str):
        img = cv2.imread(input_image)
        # 确保转换为浮点类型
        img = img.astype(np.float32)  # 添加这行
        input_tensor = np2tensor(img).unsqueeze(0)
    else:
        # 确保输入张量是浮点类型
        input_tensor = input_image.float()  # 添加这行
    
    # 添加填充操作
    input_tensor = pad_to_multiple(input_tensor, 5)
    
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    # 生成CAM图
    cam_map = score_cam(input_tensor)
    
    if cam_map is not None:
        cam_map = cam_map.cpu().numpy()
        cam_map = np.uint8(255 * cam_map)
        cam_map = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
        
        # 将热力图与原图融合
        if isinstance(input_image, str):
            result = cv2.addWeighted(img, 0.7, cam_map, 0.3, 0)
        else:
            result = cv2.addWeighted(
                tensor2np(input_tensor.squeeze(0)), 
                0.7, 
                cam_map, 
                0.3, 
                0
            )
        
        return result, cam_map
    
    return None, None

if __name__ == '__main__':
    # 示例使用
    model_path = '../../ckpt/MMBSN_SIDD_o_a45.pth'
    test_image = '../../dataset/test_data/0001_18.png'
    
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