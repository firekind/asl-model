from typing import Callable

import albumentations as A
import numpy as np
from athena.datasets import BaseDataset
from athena.utils.transforms import ToNumpy, ToTensor
from torchvision.datasets.folder import ImageFolder, default_loader


class ASLSignLanguage(ImageFolder, BaseDataset):
    url = "https://www.kaggle.com/grassknoted/asl-alphabet"
    mean = (0.5190, 0.4992, 0.5140)
    std = (0.2045, 0.2334, 0.2419)

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform: Callable = None,
        target_transform: Callable = None,
        loader: Callable = default_loader,
        use_default_transforms: bool = False,
    ):
        ImageFolder.__init__(
            self, f"{root}/asl_alphabet_train", transform, target_transform, loader
        )
        BaseDataset.__init__(
            self, root, True, transform, target_transform, False, use_default_transforms
        )

    def __getitem__(self, index):
        path: str
        target: int
        path, target = self.samples[index]

        img = np.array(self.loader(path), dtype=np.float32) / 255  # shape: (H, W, C)

        if self.transform is not None:
            if isinstance(
                self.transform, (A.BasicTransform, A.core.composition.BaseCompose)
            ):
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def default_train_transform(self) -> Callable:
        return A.Compose(
            [
                A.Lambda(ToNumpy, name="ToNumpy"),
                A.Normalize(
                    mean=ASLSignLanguage.mean,
                    std=ASLSignLanguage.std,
                    max_pixel_value=1.0,
                ),
                A.Resize(64, 64),
                A.Rotate((-10, 10)),
                A.OneOf([A.RGBShift(), A.ToGray()]),
                A.Lambda(ToTensor, name="ToTensor"),
            ]
        )

    def default_val_transform(self) -> Callable:
        return A.Compose(
            [
                A.Lambda(ToNumpy, name="ToNumpy"),
                A.Normalize(
                    mean=ASLSignLanguage.mean,
                    std=ASLSignLanguage.std,
                    max_pixel_value=1.0,
                ),
                A.Resize(64, 64),
                A.Lambda(ToTensor, name="ToTensor"),
            ]
        )
