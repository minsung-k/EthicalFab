import os
import cv2
import glob
from PIL import Image
import numpy as np


import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SequentialSampler, random_split
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from opacus.layers import DPLSTM
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------

# insert position_interval

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_files = self.load_video_files()
        self.class_map = {class_name: i for i, class_name in enumerate(os.listdir(root_dir))}

    def load_video_files(self):
        video_files = []
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for video_name in os.listdir(class_dir):
                    video_path = os.path.join(class_dir, video_name)
                    frames = sorted(os.listdir(video_path))  # Get sorted frame names
                    video_files.append((class_name, (video_path, frames)))
        return video_files

    def __len__(self):
        # Each video contributes 10 samples, so multiply by 10
        return len(self.video_files) * 10

    def __getitem__(self, idx):
        class_label, (video_path, frames) = self.video_files[idx // 10]
        num_real_images = len(frames)
    
        # Use idx % 10 for position_interval
        position_interval = idx % 10  
    
        # Calculate number of real images to include in each segment
        images_per_interval = num_real_images // 10
    
        # Ensure images_per_interval is at least 1 for proper interval calculations
        if images_per_interval == 0:
            images_per_interval = 1
    
        # Calculate start and end indices for this interval
        start_idx = position_interval * images_per_interval
        end_idx = min(start_idx + images_per_interval, num_real_images)
    
        # Load images for this interval
        frames_tensor = torch.zeros((20, 1, 32, 32), dtype=torch.float32)
        masks = torch.zeros((20,), dtype=torch.float32)
    
        # Fill in the real images
        for i in range(start_idx, end_idx):
            img_path = os.path.join(video_path, frames[i])
            image = Image.open(img_path).convert('L')  # Load image as grayscale
            image = image.resize((32, 32))
            image_array = np.array(image) / 255.0  # Normalize to [0, 1]
            frames_tensor[i - start_idx] = torch.tensor(image_array).unsqueeze(0)  # Add channel dimension
            masks[i - start_idx] = 1  # Mark as real image
    
        # Fill remaining frames with padded images (zeros)
        num_loaded = end_idx - start_idx
        num_padded = 20 - num_loaded
        if num_padded > 0:
            frames_tensor[num_loaded:] = 0  # Padded images
            masks[num_loaded:] = 0  # Mark as padded images
    
        # Prepare labels, max_seq, and position interval
        label = self.class_map[class_label]  # Use the mapping to get the integer label
        max_seq = num_real_images
    
        return frames_tensor, label, masks, max_seq, position_interval



#------------------------------------------------------------------------------
