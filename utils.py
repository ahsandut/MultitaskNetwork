import torch
from torchsummary import summary
from thop import profile
from model import MultiTaskKneeNet

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

def compute_flops(model, input_size=(1, 320, 320)):
    x = torch.randn(1, *input_size)
    flops, params = profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
    print(f"Parameters: {params/1e6:.2f} M")

def model_summary(model, input_size=(1, 320, 320)):
    summary(model, input_size=input_size)