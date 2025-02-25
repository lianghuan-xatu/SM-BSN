import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """GradCAM实现类，用于生成和可视化类激活图
    
    该类实现了GradCAM算法，可用于分析深度神经网络中的特定层对输出的贡献。
    主要用于可视化模型的决策过程和理解模型的注意力机制。
    
    Attributes:
        model: 要分析的神经网络模型
        target_layer: 目标层，用于提取特征图和梯度
        feature_maps: 保存的特征图
        gradients: 保存的梯度信息
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # 注册hook函数
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """前向传播hook，保存特征图"""
        self.feature_maps = output

    def _backward_hook(self, module, grad_input, grad_output):
        """反向传播hook，保存梯度"""
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_category=None):
        """生成CAM热力图
        
        Args:
            input_tensor: 输入张量
            target_category: 目标类别（对于分类任务）
            
        Returns:
            cam: 归一化的CAM热力图
        """
        # 前向传播
        model_output = self.model(input_tensor)

        if target_category is None:
            target_category = model_output.argmax(dim=1)

        # 清除之前的梯度
        self.model.zero_grad()

        # 计算梯度
        if len(model_output.shape) == 4:  # 对于非分类任务（如去噪）
            loss = model_output.sum()
        else:  # 对于分类任务
            loss = model_output[0, target_category]
        loss.backward(retain_graph=True)

        # 获取特征图的权重
        weights = F.adaptive_avg_pool2d(self.gradients, 1)

        # 生成CAM
        cam = torch.mul(self.feature_maps, weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU激活

        # 归一化
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

    def visualize_cam(self, input_tensor, original_image, target_category=None):
        """可视化CAM热力图
        
        Args:
            input_tensor: 输入张量
            original_image: 原始图像（numpy数组，BGR格式）
            target_category: 目标类别
            
        Returns:
            visualization: 热力图与原图的叠加结果
        """
        cam = self.generate_cam(input_tensor, target_category)
        cam = cam[0, 0]  # 取第一个样本的热力图

        # 将热力图转换为彩色图像
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # 将热力图与原图叠加
        visualization = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)
        
        return visualization