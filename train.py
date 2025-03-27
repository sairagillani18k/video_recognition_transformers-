import torch
import torch.optim as optim
import torch.nn as nn
from models.vivit import ViViT
from data.dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
train_loader, test_loader = get_dataloader()

# Initialize Model
model = ViViT(num_classes=101).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
for epoch in range(5):
    model.train()
    running_loss = 0.0

    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

torch.save(model.state_dict(), "vivit_ucf101.pth")
print("Training complete, model saved.")
