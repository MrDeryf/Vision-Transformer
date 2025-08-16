import torch
import torch.nn as nn
import wandb
from data import get_data
from model import ViT
from torch.utils.data import DataLoader
from training import test_loop, train_loop

wandb.login()
img_size = 225
patch_size = 15
train_data, test_data = get_data(img_size=img_size)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

device = "cuda"
vit_model = ViT(
    img_size=img_size,
    patch_size=patch_size,
    num_classes=101,
    drop_rate=0.3,
    num_heads=4,
)
vit_model.to(device)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.SGD(vit_model.parameters(), lr=learning_rate, momentum=0.8)

wandb.init(
    # Set the project where this run will be logged
    project="Vit_project_1",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name="experiment_1",
)

epochs = 30
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_loader, vit_model, loss_fn, optimizer, device)
    test_loop(test_loader, vit_model, loss_fn, device)

torch.save(vit_model.state_dict(), "model_weights.pth")

wandb.finish()
print("Done!")
