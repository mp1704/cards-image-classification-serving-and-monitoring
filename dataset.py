from __future__ import annotations
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as transforms

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir: str|os.PathLike, transform: Optional[torchvision.transforms] = None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.data = ImageFolder(self.data_dir, transform = transform) 
    def __len__(self): return len(self.data)
    def __getitem__(self, idx:tuple[int, int]): return self.data[idx]
    
    @property
    def classes(self): return self.data.classes