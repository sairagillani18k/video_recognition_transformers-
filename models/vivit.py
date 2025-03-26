import torch
import torch.nn as nn
import timm

class ViViT(nn.Module):
    def __init__(self, num_classes=101):
        super(ViViT, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
        
        # Adjust input layer to handle video frames (3D input)
        self.model.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = ViViT(num_classes=101)
    print(model)
