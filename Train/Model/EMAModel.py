import copy
import torch

class EMAModel:
    def __init__(self, student_model, alpha=0.999):
        self.base_alpha = alpha
        self.teacher = copy.deepcopy(student_model)
        
        self.teacher.train() 
       
        for m in self.teacher.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()  
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student_model, global_step):
        alpha = min(1.0 - 1.0 / (global_step + 1.0), self.base_alpha)
        
        for t_param, s_param in zip(self.teacher.module.parameters(), student_model.module.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)

    def __call__(self, *args, **kwargs):
        return self.teacher(*args, **kwargs)