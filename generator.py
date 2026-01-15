import torch
import torch.nn as nn
from basicsr.archs.rrdbnet_arch import RRDBNet as RRDBModel
class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        self.model = RRDBModel(
            num_in_ch=in_nc,        
            num_out_ch=out_nc,      
            num_feat=nf,            
            num_block=nb,           
            num_grow_ch=gc          
        )
    def forward(self, x):
        return self.model(x)
