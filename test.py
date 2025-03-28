import torch
from models.vivit import ViViT
from data.dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
_, test_loader = get_dataloader(batch_size=1)

# Load Model
model = ViViT(num_classes=101).to(device)
model.load_state_dict(torch.load("vivit_ucf101.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for videos, labels in test_loader:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
