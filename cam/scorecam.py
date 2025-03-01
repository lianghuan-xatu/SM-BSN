import torch
import torch.nn.functional as F
from cam.basecam import *


class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
    
        # predication on raw input
        logit = self.model_arch(input).cuda()
        
        # 修改这部分以适应去噪模型的输出
        if class_idx is None:
            # 将输出reshape为 [batch_size, -1]
            logit_flat = logit.reshape(b, -1)
            max_val, predicted_class = logit_flat.max(1)
            score = max_val.mean()  # 使用均值得到标量
        else:
            predicted_class = torch.LongTensor([class_idx]).cuda()
            logit_flat = logit.reshape(b, -1)
            score = logit_flat[:, :1].mean()  # 只取第一个元素
    
        # 不需要对图像输出做softmax
        # logit = F.softmax(logit)
    
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()
    
        score_saliency_map = torch.zeros((1, 1, h, w)).cuda()
    
        with torch.no_grad():
            for i in range(k):
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    
                if saliency_map.max() == saliency_map.min():
                    continue
    
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    
                # 修改这部分以适应去噪模型
                output = self.model_arch(input * norm_saliency_map)
                output_flat = output.reshape(b, -1)
                score = output_flat.mean()  # 使用平均值作为分数
    
                score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(
            score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)