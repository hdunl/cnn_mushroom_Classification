import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.amp import GradScaler, autocast
import time
import json
import logging
import argparse
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MushroomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(root_dir)
        self.images, self.labels = self._load_images()
        self._check_labels()
        logger.info(f"Dataset initialized with {len(self.images)} images across {len(self.classes)} classes.")
        logger.info(f"Class to index mapping: {self.class_to_idx}")
        logger.info(f"Label range: min={min(self.labels)}, max={max(self.labels)}")

    def _check_labels(self):
        num_classes = len(self.classes)
        invalid_labels = [label for label in self.labels if label < 0 or label >= num_classes]
        if invalid_labels:
            logger.error(f"Found {len(invalid_labels)} invalid labels. First few: {invalid_labels[:5]}")
            raise ValueError("Invalid labels found in dataset")
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _load_images(self):
        images = []
        labels = []
        for target_class in tqdm(self.classes, desc="Loading image paths"):
            class_dir = os.path.join(self.root_dir, target_class)
            class_idx = self.class_to_idx[target_class]
            for root, _, fnames in os.walk(class_dir):
                for fname in fnames:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(root, fname)
                        images.append(path)
                        labels.append(class_idx)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        return image, label
def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform
def load_data(data_dir, batch_size, val_split=0.2, num_workers=8):
    logger.info("Initializing datasets...")
    train_transform, val_transform = get_transforms()
    full_dataset = MushroomDataset(data_dir, transform=train_transform)
    
    classes = sorted(full_dataset.class_to_idx.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    full_dataset.class_to_idx = class_to_idx
    full_dataset.labels = [class_to_idx[full_dataset.classes[label]] for label in full_dataset.labels]
    
    logger.info(f"Updated class to index mapping: {full_dataset.class_to_idx}")
    logger.info(f"Updated label range: min={min(full_dataset.labels)}, max={max(full_dataset.labels)}")
    
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = val_transform
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True)
    logger.info(f"Train set size: {len(train_indices)}, Validation set size: {len(val_indices)}")
    return train_loader, val_loader, classes
    
class HierarchicalModel(nn.Module):
    def __init__(self, num_classes, num_groups=7):
        super(HierarchicalModel, self).__init__()
        self.base_model = models.efficientnet_v2_l(weights='IMAGENET1K_V1')
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.classes_per_group = (num_classes + num_groups - 1) // num_groups
        for param in self.base_model.parameters():
            param.requires_grad = True
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        self.group_classifier = nn.Linear(num_features, num_groups)
        self.class_classifiers = nn.ModuleList([nn.Linear(num_features, self.classes_per_group) for _ in range(num_groups)])
    
    def forward(self, x):
        features = self.base_model(x)
        group_logits = self.group_classifier(features)
        class_logits = torch.cat([classifier(features) for classifier in self.class_classifiers], dim=1)
        return group_logits, class_logits[:, :self.num_classes]  # trim excess classes

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, rank, world_size, num_classes, patience=5, checkpoint_dir='checkpoints', gradient_accumulation_steps=4):
    os.makedirs(checkpoint_dir, exist_ok=True)
    scaler = torch.amp.GradScaler()
    best_val_acc = 0
    epochs_no_improve = 0
    if rank == 0:
        logger.info(f"Starting training on device: {device}")
        logger.info(f"Number of training batches: {len(train_loader)}")
        logger.info(f"Number of validation batches: {len(val_loader)}")
    num_groups = model.module.num_groups
    classes_per_group = model.module.classes_per_group
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_loader.sampler.set_epoch(epoch)
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training', disable=rank != 0)
        optimizer.zero_grad()
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            if torch.any(labels < 0) or torch.any(labels >= num_classes):
                logger.error(f"Invalid labels detected: min={labels.min().item()}, max={labels.max().item()}, num_classes={num_classes}")
                raise ValueError("Invalid labels detected")
            with autocast(device_type='cuda', enabled=True):
                group_logits, class_logits = model(inputs)
                group_labels = labels // classes_per_group
                group_labels = torch.clamp(group_labels, max=num_groups-1)  # ensure group labels are within range
                if torch.any(group_labels < 0) or torch.any(group_labels >= num_groups):
                    logger.error(f"Invalid group labels: min={group_labels.min().item()}, max={group_labels.max().item()}, num_groups={num_groups}")
                    raise ValueError("Invalid group labels detected")
                loss = criterion(class_logits, labels) + 0.1 * criterion(group_logits, group_labels)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item() * inputs.size(0) * gradient_accumulation_steps
            _, predicted = class_logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            if rank == 0:
                train_bar.set_postfix({
                    'loss': f'{train_loss / (batch_idx + 1):.4f}',
                    'acc': f'{train_correct / train_total:.4f}'
                })
        
        # validation loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation', disable=rank != 0)
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_bar):
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast(device_type='cuda', enabled=True):
                    group_logits, class_logits = model(inputs)
                    group_labels = labels // classes_per_group
                    group_labels = torch.clamp(group_labels, max=num_groups-1)
                    loss = criterion(class_logits, labels) + 0.1 * criterion(group_logits, group_labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = class_logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                if rank == 0:
                    val_bar.set_postfix({
                        'loss': f'{val_loss / (batch_idx + 1):.4f}',
                        'acc': f'{val_correct / val_total:.4f}'
                    })
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start_time
        
        if rank == 0:
            logger.info(f'Epoch {epoch+1}/{num_epochs} - '
                        f'Time: {epoch_time:.2f}s - '
                        f'Train Loss: {train_loss / len(train_loader.dataset):.4f}, Train Acc: {train_correct / train_total:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                        f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss / len(train_loader.dataset),
                'val_loss': val_loss,
                'train_acc': train_correct / train_total,
                'val_acc': val_acc,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved: {checkpoint_path}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.module.state_dict(), best_model_path)
                logger.info(f'New best model saved with validation accuracy: {val_acc:.4f}')
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve == patience:
                logger.info(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    return model
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
    
def error_analysis(model, val_loader, device, classes, top_k=5):
    model.eval()
    all_preds = []
    all_labels = []
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Performing error analysis"):
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, preds = outputs.topk(top_k, 1, True, True)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct = preds.eq(labels.view(-1, 1).expand_as(preds))
            for i, label in enumerate(labels):
                class_correct[label] += correct[i][0].item()
                class_total[label] += 1
    class_accuracy = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    top_20_classes = sorted(range(len(class_accuracy)), key=lambda i: class_accuracy[i], reverse=True)[:20]
    cm = confusion_matrix([label for label in all_labels if label in top_20_classes],
                          [pred[0] for pred, label in zip(all_preds, all_labels) if label in top_20_classes])
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[classes[i] for i in top_20_classes],
                yticklabels=[classes[i] for i in top_20_classes])
    plt.title('Confusion Matrix for Top 20 Classes')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix_top20.png')
    plt.close()
    top_10 = sorted(range(len(class_accuracy)), key=lambda i: class_accuracy[i], reverse=True)[:10]
    bottom_10 = sorted(range(len(class_accuracy)), key=lambda i: class_accuracy[i])[:10]
    plt.figure(figsize=(15, 10))
    plt.bar(range(10), [class_accuracy[i] for i in top_10], align='center')
    plt.title('Top 10 Classes by Accuracy')
    plt.xticks(range(10), [classes[i] for i in top_10], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top10_classes.png')
    plt.close()
    plt.figure(figsize=(15, 10))
    plt.bar(range(10), [class_accuracy[i] for i in bottom_10], align='center')
    plt.title('Bottom 10 Classes by Accuracy')
    plt.xticks(range(10), [classes[i] for i in bottom_10], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('bottom10_classes.png')
    plt.close()
    logger.info("Error analysis completed. Plots saved.")
    
def main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    val_split = args.val_split
    num_workers = args.num_workers
    checkpoint_dir = args.checkpoint_dir
    gradient_accumulation_steps = args.gradient_accumulation_steps

    if rank == 0:
        logger.info(f"Loading data from {data_dir}...")
        start_time = time.time()
    train_loader, val_loader, classes = load_data(data_dir, batch_size, val_split, num_workers)
    num_classes = len(classes)

    if rank == 0:
        logger.info(f"Data loading completed in {time.time() - start_time:.2f} seconds.")
        logger.info(f"Number of classes: {num_classes}")

    if rank == 0:
        logger.info("Creating model...")
    model = HierarchicalModel(num_classes=num_classes, num_groups=7).to(device)  # Set num_groups to 7
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    if rank == 0:
        logger.info("Starting training...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, rank, world_size, num_classes, checkpoint_dir=checkpoint_dir, gradient_accumulation_steps=gradient_accumulation_steps)

    if rank == 0:
        logger.info("Training completed. Performing error analysis...")
    error_analysis(model.module, val_loader, device, classes)
    cleanup()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mushroom Classification Training Script")
    parser.add_argument('--data_dir', type=str, default='/home/mushroom/all_species', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (per GPU)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num_groups', type=int, default=10, help='Number of groups for hierarchical classification')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
