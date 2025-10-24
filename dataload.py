# dataload.py

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os.path  # 确保导入了 os.path

# ImageNet 均值和标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_suffix='.jpg', mask_suffix='.png'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        # --- 关键修改：通过 "文件名" (basename) 来匹配 ---

        # 1. 获取所有图像的 "文件名" (不带.jpg)
        image_filenames = [f for f in os.listdir(self.image_dir) if f.endswith(self.image_suffix)]
        # (例如: {"353", "354", ...})
        self.image_basenames = set([os.path.splitext(f)[0] for f in image_filenames])

        # 2. 获取所有掩码的 "文件名" (不带.png或.jpg)
        mask_filenames = [f for f in os.listdir(self.mask_dir) if f.endswith(self.mask_suffix)]
        # (例如: {"353", "354", ...})
        self.mask_basenames = set([os.path.splitext(f)[0] for f in mask_filenames])

        # 3. 找到交集 (在两个文件夹中都存在的 "文件名")
        valid_basenames = self.image_basenames.intersection(self.mask_basenames)

        # 4. 存储排序后的有效 "文件名" 列表
        # (例如: ["0", "1", ..., "353", "354", ...])
        self.valid_files = sorted(list(valid_basenames))

        if len(self.valid_files) == 0:
            print(f"警告: 在 {image_dir} 和 {mask_dir} 中没有找到匹配的文件。")
            print(f"  (查找: *{image_suffix} 和 *{mask_suffix})")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, index):
        # 1. 获取 "文件名" (e.g., "353")
        basename = self.valid_files[index]

        # 2. 构建正确的路径
        img_path = os.path.join(self.image_dir, f"{basename}{self.image_suffix}")
        mask_path = os.path.join(self.mask_dir, f"{basename}{self.mask_suffix}")

        # 3. 读取 Image 和 Mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # "L" for grayscale

        # 4. 预处理Mask (二值化, 解决 .jpg 噪点 和 .png 灰阶)
        mask = (mask > 128).astype(np.uint8)

        # 5. 应用数据增强 (Albumentations)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 6. 返回，Mask需要是LongTensor
        return image, mask.long()


def get_loaders(data_path, batch_size, image_height, image_width, num_workers=4, pin_memory=True):
    """
    创建并返回训练和验证的 DataLoaders
    """
    TRAIN_IMG_DIR = os.path.join(data_path, "train/images")
    TRAIN_MASK_DIR = os.path.join(data_path, "train/masks")
    VAL_IMG_DIR = os.path.join(data_path, "val/images")
    VAL_MASK_DIR = os.path.join(data_path, "val/masks")

    # 训练集数据增强
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # 验证集数据增强 (只做 Resize 和 Normalize)
    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # --- 关键修改：传入正确的后缀 ---

    # 创建训练集 Dataset
    train_dataset = ISICDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=train_transform,
        image_suffix='.jpg',
        mask_suffix='.png'  # <-- 训练集使用 .png
    )

    # 创建验证集 Dataset
    val_dataset = ISICDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=val_transform,
        image_suffix='.jpg',
        mask_suffix='.jpg'  # <-- 验证集使用 .jpg
    )

    # 创建 DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")

    return train_loader, val_loader