import torch
import torch.nn.functional as F
from cam.basecam import *


class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)
        self.handlers = []
        self.handlers.append(
            self.target_layer.register_backward_hook(self.backward_hook)
        )
        self.handlers.append(
            self.target_layer.register_forward_hook(self.forward_hook)
        )

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    def forward(self, input, class_idx=None, retain_graph=False):
        # 限制输入批次大小
        if input.size(0) > 1:
            input = input[:1]  # 只处理第一个样本
        
        b, c, h, w = input.size()
        
        # 确保输入需要梯度
        input.requires_grad_(True)
        
        # 修改尺寸限制，确保能被5整除
        target_h = ((h - 1) // 5 + 1) * 5
        target_w = ((w - 1) // 5 + 1) * 5
        if h > 512 or w > 512:
            target_h = 510  # 确保能被5整除
            target_w = 510
        
        if h != target_h or w != target_w:
            input = F.interpolate(input, size=(target_h, target_w), mode='bilinear', align_corners=False)
            h, w = input.shape[-2:]

        # 使用 torch.cuda.amp 进行混合精度计算
        # 在前向传播部分
        with torch.cuda.amp.autocast():
            logit = self.model_arch(input)
            logit = F.softmax(logit, dim=1)  # 指定dim参数
            
            if class_idx is None:
                # 修改这里，确保得到标量值
                max_val, max_idx = logit.max(1)
                score = max_val.mean()  # 使用mean()得到标量
                predicted_class = max_idx
            else:
                predicted_class = torch.LongTensor([class_idx])
                score = logit[:, class_idx].mean()  # 使用mean()得到标量
            
            logit = F.softmax(logit)
            
            if torch.cuda.is_available():
                predicted_class = predicted_class.cuda()
                score = score.cuda()
                logit = logit.cuda()
    
        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
              
              if saliency_map.max() == saliency_map.min():
                continue
              
              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # how much increase if keeping the highlighted region
              # predication on masked input
              output = self.model_arch(input * norm_saliency_map)
              output = F.softmax(output)
              score = output[0][predicted_class]

              score_saliency_map +=  score * saliency_map
                
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)