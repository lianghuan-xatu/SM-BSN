import torch
import cv2
import numpy as np
import torch.nn.functional as F
from model.get_model import BSN
from analysis.gradcam.gradcam import GradCAM

class ReceptiveFieldAnalyzer:
    """感受野分析器，用于分析神经网络不同分支的感受野特性
    
    该类使用GradCAM技术来可视化和分析神经网络中不同分支的感受野。
    可以帮助理解网络各个分支对输入图像不同区域的关注程度。
    """
    
    def __init__(self, model_path):
        """初始化感受野分析器
        
        Args:
            model_path: 预训练模型的路径
        """
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """加载预训练的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            model: 加载好的模型
        """
        model = BSN.bsn_model()
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 处理检查点格式
        if 'model_weight' in checkpoint:
            state_dict = checkpoint['model_weight']
            if 'denoiser' in state_dict:
                state_dict = state_dict['denoiser']
        else:
            state_dict = checkpoint
            
        # 加载模型权重
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    
    def _get_target_layer(self, branch_name):
        """获取指定分支的目标层
        
        Args:
            branch_name: 分支名称，可选值：'central', 'row', 'col', 'fsz'
            
        Returns:
            target_layer: 目标层
            
        Raises:
            ValueError: 当指定了不支持的分支名称时
        """
        if branch_name == 'central':
            return self.model.bsn.branch1_1.head[0]  # 中心分支
        elif branch_name == 'row':
            return self.model.bsn.branch3_1.head[0]  # 行分支
        elif branch_name == 'col':
            return self.model.bsn.branch2_1.head[0]  # 列分支
        # elif branch_name == 'fsz':
        #     return self.model.bsn.branch5_1.head[0]  # 全尺寸分支
        else:
            raise ValueError(f'不支持的分支类型: {branch_name}')
    
    def _pad_to_multiple(self, x, factor):
        """将张量填充到能被factor整除的尺寸
        
        Args:
            x: 输入张量
            factor: 填充因子
            
        Returns:
            padded_x: 填充后的张量
        """
        h, w = x.shape[-2:]
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x
    
    def analyze_branch(self, input_image, branch_name):
        """分析特定分支的感受野
        
        Args:
            input_image: 输入图像（numpy数组，BGR格式）
            branch_name: 要分析的分支名称
            
        Returns:
            visualization: 感受野可视化结果
        """
        # 获取目标层
        target_layer = self._get_target_layer(branch_name)
        
        # 初始化GradCAM
        grad_cam = GradCAM(self.model, target_layer)
        
        # 准备输入
        input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float()
        # 填充到能被5整除的尺寸
        input_tensor = self._pad_to_multiple(input_tensor, 5)
        input_tensor.requires_grad = True
        
        # 生成热力图
        visualization = grad_cam.visualize_cam(input_tensor, input_image)
        
        return visualization

def main():
    """主函数，用于演示感受野分析的使用方法"""
    # 初始化分析器
    # 使用相对于项目根目录的路径
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(root_dir, 'ckpt', 'MMBSN_SIDD_o_a45.pth')
    analyzer = ReceptiveFieldAnalyzer(model_path)
    
    # 加载测试图像
    image_path = os.path.join(root_dir, 'dataset', 'test_data', '0001_18.png')
    image = cv2.imread(image_path)
    
    # 分析不同分支的感受野
    branches = ['central', 'row', 'col', 'fsz']
    for branch in branches:
        vis = analyzer.analyze_branch(image, branch)
        
        # 保存结果
        output_path = f'output/receptive_field_{branch}.png'
        cv2.imwrite(output_path, vis)
        print(f'已保存{branch}分支的感受野分析结果到: {output_path}')

if __name__ == '__main__':
    main()