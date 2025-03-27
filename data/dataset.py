import torch
import torchvision.transforms as transforms
from torchvision.datasets import UCF101
from torch.utils.data import DataLoader

def get_dataloader(batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = UCF101(root="./data", annotation_path="./data/ucf101.json", frames_per_clip=16, step_between_clips=1, train=True, transform=transform, download=True)
    test_dataset = UCF101(root="./data", annotation_path="./data/ucf101.json", frames_per_clip=16, step_between_clips=1, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
