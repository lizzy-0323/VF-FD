import torch
import torchvision

device = torch.device("mps")

x = torch.randn(128, 128, device=device)
net = torchvision.models.resnet18().to(device)

print(x.device)
print(next(net.parameters()).device)
