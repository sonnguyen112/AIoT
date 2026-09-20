"""Fine-tune a ResNet50 to classify dodo and no_dodo images.

Expected dataset layout::

    midterm/
      train.py
      dataset/
        dodo/
        no_dodo/

Example:
    python train.py --epochs 15

The best checkpoint is written to ``midterm/checkpoints/best_resnet50.pth``
unless ``--output-dir`` is provided.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import torch
from PIL import ImageFile
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_CLASSES = {"dodo", "no_dodo"}


class BinaryImageFolder(datasets.ImageFolder):
    """ImageFolder view that ignores auxiliary folders such as ``dataset/test``."""

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        classes = sorted(
            class_name
            for class_name in EXPECTED_CLASSES
            if (Path(directory) / class_name).is_dir()
        )
        if set(classes) != EXPECTED_CLASSES:
            raise ValueError(
                f"Expected class folders {sorted(EXPECTED_CLASSES)} directly under {directory}."
            )
        return classes, {class_name: index for index, class_name in enumerate(classes)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained ResNet50 for dodo/no_dodo classification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "dataset",
        help="Folder containing one subfolder per class (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "checkpoints",
        help="Folder for checkpoints and training history.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4, help="Backbone learning rate.")
    parser.add_argument(
        "--head-lr", type=float, default=1e-3, help="Learning rate for the replacement classifier."
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=1,
        help="Train only the new classifier for this many epochs before full fine-tuning.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop after this many validation epochs without improvement.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download or use ImageNet pretrained weights.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Make data splitting and training as repeatable as possible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    """Choose CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_split(
    targets: Iterable[int], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split indices while keeping every class in train and validation sets."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")

    targets = list(targets)
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(target, []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_indices in by_class.values():
        if len(class_indices) < 2:
            raise ValueError("Each class needs at least two images for a train/validation split.")
        shuffled = class_indices[:]
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_ratio))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    return sorted(train_indices), sorted(val_indices)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return stronger augmentation for training and deterministic validation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Create stratified train and validation data loaders."""
    train_transform, val_transform = build_transforms()
    train_dataset = BinaryImageFolder(args.data_dir, transform=train_transform)
    val_dataset = BinaryImageFolder(args.data_dir, transform=val_transform)

    available_classes = set(train_dataset.classes)
    if available_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected class folders {sorted(EXPECTED_CLASSES)}, "
            f"found {train_dataset.classes} in {args.data_dir}."
        )
    if train_dataset.samples != val_dataset.samples:
        raise RuntimeError("Train and validation datasets do not contain the same files.")

    train_indices, val_indices = stratified_split(
        train_dataset.targets, val_ratio=args.val_ratio, seed=args.seed
    )
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_dataset.class_to_idx


def build_model(num_classes: int, use_pretrained: bool) -> nn.Module:
    """Load ResNet50 and replace its ImageNet classifier."""
    weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze every layer except the replacement classifier."""
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, float]:
    """Run one train or validation epoch and return loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise RuntimeError("The data loader is empty.")
    return total_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> None:
    """Save everything needed to resume or run inference later."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "class_to_idx": class_to_idx,
        "image_size": IMAGE_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("--freeze-backbone-epochs cannot be negative.")

    seed_everything(args.seed)
    device = choose_device()
    train_loader, val_loader, class_to_idx = build_dataloaders(args)
    model = build_model(len(class_to_idx), use_pretrained=not args.no_pretrained).to(device)

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, trainable=False)
    else:
        set_backbone_trainable(model, trainable=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("fc.")
                ],
                "lr": args.lr,
            },
            {"params": model.fc.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_resnet50.pth"
    last_path = args.output_dir / "last_resnet50.pth"
    history_path = args.output_dir / "history.json"

    print(f"Device: {device}")
    print(f"Classes: {class_to_idx}")
    print(f"Train images: {len(train_loader.dataset)} | Validation images: {len(val_loader.dataset)}")

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            set_backbone_trainable(model, trainable=True)

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[-1]["lr"],
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.3f}"
        )

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss,
            val_accuracy,
            class_to_idx,
            args,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                val_accuracy,
                class_to_idx,
                args,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
