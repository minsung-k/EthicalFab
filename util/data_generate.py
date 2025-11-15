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
    """
    A dataset that loads sequences of video frames for classification tasks.

    Each directory in ``root_dir`` is assumed to correspond to a class name and
    contains sub‑directories for individual videos. Each video directory holds
    individual frame files. For every video we generate ``intervals`` samples by
    partitioning the sequence into equal segments. Within each segment we take
    a subset of frames and pad/truncate them to a fixed length ``frames_per_video``.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root. Should contain one subdirectory per class.
    frames_per_video : int, optional
        Fixed number of frames returned for each sample. Default: 20.
    intervals : int, optional
        Number of segments to divide each video into. This determines how many
        samples each video contributes. Default: 10.
    image_size : tuple of int, optional
        Target size ``(height, width)`` for resizing frames. Default: (32, 32).
    transform : callable, optional
        Optional transform to be applied on a frame (after resizing and
        normalisation). It should accept a PIL.Image and return a tensor.
    cache : bool, optional
        If True, decoded and resized frames will be cached in memory the
        first time they are accessed. This can drastically speed up
        subsequent accesses at the cost of higher memory usage.

    Returns
    -------
    tuple
        A tuple ``(frames, label, mask, max_seq, position_interval)`` where

        * ``frames`` is a tensor of shape ``(frames_per_video, 1, H, W)``;
        * ``label`` is an integer index of the class;
        * ``mask`` is a 1D tensor of length ``frames_per_video`` indicating
          which entries in ``frames`` are real frames (1) and which are
          padding (0);
        * ``max_seq`` is the original number of frames in the video;
        * ``position_interval`` is the zero‑based index of the segment within
          the video (from 0 to ``intervals - 1``).
    """

    def __init__(self,
                 root_dir: str,
                 frames_per_video: int = 20,
                 intervals: int = 10,
                 image_size: tuple = (32, 32),
                 transform: callable = None,
                 cache: bool = False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video
        self.intervals = intervals
        self.image_size = image_size
        self.transform = transform
        self.cache = cache

        # Build class to index mapping, sorting to ensure deterministic ordering
        class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        class_names.sort()
        self.class_map = {name: i for i, name in enumerate(class_names)}

        # Build list of videos. Each entry is (class_name, [frame_path1, frame_path2, ...]).
        self.video_files = []
        for class_name in class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            for video_name in sorted(os.listdir(class_dir)):
                video_path = os.path.join(class_dir, video_name)
                if not os.path.isdir(video_path):
                    continue
                # Collect all image files with common extensions
                frames = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))
                          if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]
                if frames:
                    self.video_files.append((class_name, frames))

        # Optionally cache decoded frames. The cache key is the frame path.
        self._frame_cache = {} if cache else None

    def __len__(self) -> int:
        # Each video contributes ``intervals`` samples
        return len(self.video_files) * self.intervals

    def _load_frame(self, path: str) -> torch.Tensor:
        """Load an image from disk, convert to grayscale and resize to ``image_size``.

        If caching is enabled the decoded result will be stored and returned
        directly on subsequent calls.
        """
        if self._frame_cache is not None and path in self._frame_cache:
            return self._frame_cache[path]

        # Read the image. Use PIL for consistent behaviour across platforms.
        with Image.open(path) as img:
            img = img.convert('L')  # Convert to grayscale
            if self.image_size is not None:
                img = img.resize(self.image_size, Image.BILINEAR)
            # Apply transform if provided. The transform is responsible for
            # converting the PIL image into a tensor (e.g. via ToTensor).
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                # Convert to numpy array and normalise to [0, 1]
                arr = np.asarray(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).unsqueeze(0)  # add channel dimension

        if self._frame_cache is not None:
            self._frame_cache[path] = tensor
        return tensor

    def __getitem__(self, idx: int):
        # Determine which video and which interval this index corresponds to
        video_idx = idx // self.intervals
        position_interval = idx % self.intervals
        class_name, frame_paths = self.video_files[video_idx]
        num_real_images = len(frame_paths)

        # Determine how many frames belong to each interval. Use ceil division to
        # ensure the entire sequence is covered. This also prevents zero frames
        # when ``num_real_images`` < ``intervals``.
        images_per_interval = max((num_real_images + self.intervals - 1) // self.intervals, 1)
        start_idx = position_interval * images_per_interval
        end_idx = min(start_idx + images_per_interval, num_real_images)

        # Allocate tensors for the sequence and mask
        frames_tensor = torch.zeros((self.frames_per_video, 1, *self.image_size), dtype=torch.float32)
        mask = torch.zeros((self.frames_per_video,), dtype=torch.float32)

        # Fill the sequence tensor with real frames
        load_count = 0
        for i in range(start_idx, end_idx):
            if load_count >= self.frames_per_video:
                break  # In case more frames than frames_per_video
            frame_tensor = self._load_frame(frame_paths[i])
            if frame_tensor.dim() == 3:
                # Already (C, H, W)
                frames_tensor[load_count] = frame_tensor
            else:
                # If transform returned 2D (H, W), unsqueeze channel
                frames_tensor[load_count] = frame_tensor.unsqueeze(0)
            mask[load_count] = 1.0
            load_count += 1

        # ``max_seq`` indicates the original number of frames. Useful for
        # sequence models that rely on variable lengths.
        label = self.class_map[class_name]
        max_seq = num_real_images

        return frames_tensor, label, mask, max_seq, position_interval



#------------------------------------------------------------------------------
