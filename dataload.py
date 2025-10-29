import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os.path

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

        # (这部分文件匹配逻辑不变)
        image_filenames = [f for f in os.listdir(self.image_dir) if f.endswith(self.image_suffix)]
        self.image_basenames = set([os.path.splitext(f)[0] for f in image_filenames])
        mask_filenames = [f for f in os.listdir(self.mask_dir) if f.endswith(self.mask_suffix)]
        self.mask_basenames = set([os.path.splitext(f)[0] for f in mask_filenames])
        valid_basenames = self.image_basenames.intersection(self.mask_basenames)
        self.valid_files = sorted(list(valid_basenames))

        if len(self.valid_files) == 0:
            print(f"警告: 在 {image_dir} 和 {mask_dir} 中没有找到匹配的文件。")
            print(f"  (查找: *{image_suffix} 和 *{mask_suffix})")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, index):
        basename = self.valid_files[index]
        img_path = os.path.join(self.image_dir, f"{basename}{self.image_suffix}")
        mask_path = os.path.join(self.mask_dir, f"{basename}{self.mask_suffix}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # 预处理Mask (二值化)
        mask = (mask > 128).astype(np.uint8)

        # 应用数据增强 (Albumentations)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']  # mask 此时是 [H, W], LongTensor

        # <-- 修改点: 转换为 [1, H, W] 的 FloatTensor
        # .unsqueeze(0) 在 0 维增加通道
        # .float() 转换为浮点型以匹配BCE Loss的需要
        return image, mask.unsqueeze(0).float()


# (get_loaders 函数保持不变)
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

    # # 验证集数据增强
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

    # (Dataset 的调用不变)
    train_dataset = ISICDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=train_transform,
        image_suffix='.jpg',
        mask_suffix='.png'
    )

    val_dataset = ISICDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=val_transform,
        image_suffix='.jpg',
        mask_suffix='.jpg'
    )

    # (DataLoader 的调用不变)
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