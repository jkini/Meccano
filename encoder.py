import torch
import torch.nn as nn
import torchvision.models.video.swin_transformer as swin
from ss_swin import SwinTransformer3D
import torch.nn.functional as F

class SSModel(nn.Module):
    def __init__(self, num_classes):
        super(SSModel, self).__init__()
        self.swin3d_b_ss = SwinTransformer3D(embed_dim=128, 
                          depths=[2, 2, 18, 2], 
                          num_heads=[4, 8, 16, 32], 
                          patch_size=(2,4,4), 
                          window_size=(16,7,7), 
                          drop_path_rate=0.4, 
                          patch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.linear = nn.Linear(self.swin3d_b_ss.layers[-1].blocks[-1].mlp.fc2.out_features, num_classes)

    def forward(self, x):
        mm_model = self.swin3d_b_ss(x)
        mm_pool = self.avgpool(mm_model)
        mm_pool = mm_pool.view(mm_pool.size(0), -1)
        mm_fc = self.linear(mm_pool)
        return mm_fc



