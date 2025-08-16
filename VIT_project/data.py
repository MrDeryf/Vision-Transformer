from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms


def get_data(data_dir: str = "data", img_size: int = 224):
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data = datasets.Caltech101(root=data_dir, download=True, transform=transform)

    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])

    return train_data, test_data
