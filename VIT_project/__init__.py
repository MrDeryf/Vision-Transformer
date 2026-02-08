import torch
import wandb
from data import get_data
from model import ViT
from torch.utils.data import DataLoader
from training import test_loop, train_loop
import config

wandb.login()
img_size = config.IMG_SIZE
patch_size = config.PATCH_SIZE
train_data, test_data = get_data(img_size=img_size)

batch_size = config.BATCH_SIZE
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = ("cuda" if torch.cuda.is_available() else "cpu")
vit_model = ViT(
    img_size=img_size,
    patch_size=patch_size,
    num_classes=config.NUM_OF_CLASSES,
    drop_rate=config.DROP_RATE,
    num_heads=config.NUM_OF_HEADS,
)
vit_model.to(device)

loss_fn = config.LOSS_FN()
optimizer = config.OPRIMIZER(vit_model.parameters(),
                             lr=config.LEARNING_RATE,
                             momentum=config.MOMENTUM)

wandb.init(
    # Set the project where this run will be logged
    project="Vit_project_1",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name="experiment_1",
)

epochs = config.EPOCHS
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_loader, vit_model, loss_fn, optimizer, device)
    test_loop(test_loader, vit_model, loss_fn, device)

torch.save(vit_model.state_dict(), "model_weights.pth")

wandb.finish()
print("Done!")
