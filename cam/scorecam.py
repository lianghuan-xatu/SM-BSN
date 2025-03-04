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
        
        # - max(1) 中的参数 1 表示在哪个维度上求最大值：
        # - dim=0 ：在第一个维度（行）
        # - dim=1 ：在第二个维度（列）
        # - max() 函数返回两个值的元组：
        # - [0] ：最大值
        # - [1] 或 [-1] ：最大值的索引位置
        # 修改这部分代码
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        # 添加 sum() 使其变为标量
        score = score.sum()  # 确保是标量值

        # 指定 dim=1 进行 softmax
        logit = F.softmax(logit, dim=1)

        if torch.cuda.is_available():
            predicted_class = predicted_class.cuda()
            score = score.cuda()
            logit = logit.cuda()

        self.model_arch.zero_grad()
        # retain_graph 参数决定是否保留计算图，用于多次反向传播
        score.backward(retain_graph=retain_graph)
        # 这些激活值是通过钩子（hook）在前向传播时保存的
        activations = self.activations['value']
        b, k, u, v = activations.size()
        
        # score_saliency_map 是用来存储和累积显著性图（注意力图）的张量
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          # k是特征图的数量
          for i in range(k):

              # upsampling
              # - 从激活层中选择第 i 个特征图： activations[:, i, :, :] 
              # unsqueeze(1) 在第1维增加一个维度
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
              
              # 如果最大值等于最小值，说明：
                # - 特征图中所有像素值都相同
                # - 这样的特征图没有任何区分度
                # - 对最终的注意力图没有贡献
              if saliency_map.max() == saliency_map.min():
                continue
              
              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # how much increase if keeping the highlighted region
              # predication on masked input
              output = self.model_arch(input * norm_saliency_map)
              output = F.softmax(output, dim=1)
              score = output[0][predicted_class]
              
              # 修复维度问题
              if isinstance(score, torch.Tensor):
                  if score.numel() > 1:  # 如果张量包含多个元素
                      score = score.mean()  # 取平均值
                  score = score.item()  # 将单个元素张量转换为标量
              
              # 创建正确形状的张量
              score = torch.tensor([[[[score]]]], device=saliency_map.device)
              score_saliency_map += score * saliency_map

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