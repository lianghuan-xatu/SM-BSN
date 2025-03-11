import torch.nn.functional as F
from analysis.scorecam.basecam import *


class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        # 添加批处理机制
        batch_size = 32
        b, c, h, w = input.size()
        
        # 获取模型输出
        output = self.model_arch(input).cuda()
        score = torch.mean(torch.abs(output - input))
        
        if torch.cuda.is_available():
            score = score.cuda()
    
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        
        # 获取特征图
        activations = self.activations['value']
        b, k, u, v = activations.size()
        
        # MM-BSN中使用了5x5的像素洗牌，需要先进行反向像素洗牌操作
        pixel_shuffle = 5
        # 计算填充通道
        total_channels = k * (pixel_shuffle ** 2)
        activations = activations.view(b, k, 1, 1, u, v)
        activations = activations.expand(b, k, pixel_shuffle, pixel_shuffle, u, v)
        activations = activations.contiguous().view(b, total_channels, u, v)
        
        # 应用反向像素洗牌
        activations = F.pixel_unshuffle(activations, pixel_shuffle)
        
        score_saliency_map = torch.zeros((1, 1, h, w)).cuda()
        weights = []
        saliency_maps = []
    
        with torch.no_grad():
            # 处理每个特征图
            for i in range(activations.size(1)):
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                # 直接使用原始尺寸，避免额外的插值操作
                if saliency_map.size(2) != h:
                    saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                
                if saliency_map.max() == saliency_map.min():
                    continue
                
                # 归一化特征图
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                
                # 使用去噪效果作为权重
                masked_input = input * norm_saliency_map
                masked_output = self.model_arch(masked_input)
                weight = -torch.mean(torch.abs(masked_output - input))
                
                weights.append(weight.item())
                saliency_maps.append(saliency_map)
    
            # 生成最终的CAM图
            if weights:
                weights = F.softmax(torch.tensor(weights, device=score_saliency_map.device), dim=0)
                for weight, smap in zip(weights, saliency_maps):
                    score_saliency_map += weight * smap

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        # 检查是否是平坦图
        if score_saliency_map_min == score_saliency_map_max:
            return None
        # 最终归一化
        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)