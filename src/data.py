import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import lightning as L
import logging

logger = logging.getLogger(__name__)


class BalancedBinaryDataset(torch.utils.data.Dataset):
    """
    Dataset that balances two subsets by oversampling the smaller one.
    
    For OOD detection, this ensures equal representation of ID and OOD-proxy
    classes during training.
    """
    
    def __init__(self, id_dataset, ood_dataset):
        self.id_data = id_dataset
        self.ood_data = ood_dataset
        self.id_len = len(id_dataset)
        self.ood_len = len(ood_dataset)
        
        # Balance by matching the larger class size
        self.total_len = max(self.id_len, self.ood_len) * 2
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx % 2 == 0:
            img, _ = self.id_data[(idx // 2) % self.id_len]
            return img, 0
        else:
            img, _ = self.ood_data[(idx // 2) % self.ood_len]
            return img, 1


class CIFAR10BinaryDataModule(L.LightningDataModule):
    """
    CIFAR-10 data module with binary class split for OOD detection.
    
    Splits data into:
        - c=0: In-distribution class (configurable, default: airplane)
        - c=1: Out-of-distribution proxy (all other classes)
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        id_class: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.id_class = id_class
        self.pin_memory = pin_memory
        
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
    
    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_train = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, transform=self.train_transform
            )
            full_val = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, transform=self.val_transform
            )
            
            train_id_idx = [i for i, (_, label) in enumerate(full_train) if label == self.id_class]
            train_ood_idx = [i for i, (_, label) in enumerate(full_train) if label != self.id_class]
            
            self.train_id = Subset(full_train, train_id_idx)
            self.train_ood = Subset(full_train, train_ood_idx)
            self.val_dataset = full_val
            
            id_name = self.class_names[self.id_class]
            logger.info(f"Dataset setup complete:")
            logger.info(f"  ID class (c=0): {id_name} - {len(self.train_id)} train samples")
            logger.info(f"  OOD proxy (c=1): rest - {len(self.train_ood)} train samples")
            logger.info(f"  Validation: {len(self.val_dataset)} samples")
        
        if stage == "test" or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, transform=self.val_transform
            )
    
    def train_dataloader(self):
        dataset = BalancedBinaryDataset(self.train_id, self.train_ood)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._train_collate,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        # Create balanced validation using stratified sampling
        # This addresses the 9:1 class imbalance (ID is only 10% of test set)
        targets = torch.tensor([label for _, label in self.val_dataset])
        binary_targets = (targets != self.id_class).long()
        
        # Calculate weights for balanced sampling
        class_counts = torch.bincount(binary_targets)
        weights = 1.0 / class_counts[binary_targets].float()
        
        # Use WeightedRandomSampler for stratified validation
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.val_dataset),
            replacement=True,
        )
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._val_collate,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._val_collate,
        )
    
    def _train_collate(self, batch):
        images = torch.stack([item[0] for item in batch])
        binary_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return {"images": images, "binary_labels": binary_labels}
    
    def _val_collate(self, batch):
        images = torch.stack([item[0] for item in batch])
        original_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        binary_labels = (original_labels != self.id_class).long()
        return {
            "images": images,
            "binary_labels": binary_labels,
            "original_labels": original_labels,
        }
