import os
from typing import List, Tuple, Dict
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# CIFAR-10 normalization stats (pre-computed)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_transforms() -> Dict[str, transforms.Compose]:
    """
    Returns a dictionary of transforms for train/val/test.
    - Train: with augmentation
    - Val/Test: only basic preprocessing + normalization
    """

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    return {
        "train": train_transforms,
        "val": eval_transforms,
        "test": eval_transforms,
    }

class CIFAR10Subset(Dataset):
    """
    Wraps a CIFAR-10 dataset + a set of indices + a transform.
    This allows us to use different transforms for train vs val
    even though they come from the same original dataset.
    """
    def __init__(self, base_dataset: datasets.CIFAR10, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        base_index = self.indices[index]
        img, label = self.base_dataset[base_index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    

def get_dataloaders(
        data_dir: str = "data",
        batch_size: int = 128,
        num_workers: int = 2,
        val_size: int = 5000,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Creates train, val, and test dataloaders for CIFAR-10.

    - data_dir: where CIFAR-10 will be downloaded/cached
    - batch_size: batch size for all loaders
    - num_workers: DataLoader workers
    - val_size: number of validation samples (from the 50k train images)
    - seed: deterministic split

    Returns:
        train_loader, val_loader, test_loader, class_names
    """

    transforms_dict = get_transforms()

    # Base training dataset without transform (so we can apply different transforms per subset)
    base_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None
    )
    # Test dataset with eval/test transform
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms_dict["test"],
    )

    num_train = len(base_train)
    assert val_size < num_train, "Validation size must be less than total training size"

    # deterministic split for train/val
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_train, generator=generator).tolist()
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_dataset = CIFAR10Subset(
        base_dataset=base_train,
        indices=train_indices,
        transform=transforms_dict["train"]
    )

    val_dataset = CIFAR10Subset(
        base_dataset=base_train,    
        indices=val_indices,
        transform=transforms_dict["val"]
    )

    # Create DataLoaders    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_names = base_train.classes
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    """
    Quick sanity check:
    Run:  python -m src.dataset   (from project root)
    """
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir="data",
        batch_size=64,
        num_workers=2,
        val_size=5000,
    )

    print("Classes:", class_names)
    example_batch = next(iter(train_loader))
    images, labels = example_batch
    print("Train batch shape:", images.shape)   
    print("Train labels shape:", labels.shape) 