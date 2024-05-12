import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np


class MNISTDataset(Dataset):
    def __init__(self, image_filename: str, label_filename: str, transforms: Optional[List] = None):
        super().__init__(transforms)
        with gzip.open(image_filename, 'rb') as f:
            _, num_img, row, col = struct.unpack('>IIII', f.read(16))
            X = np.frombuffer(f.read(), np.uint8).byteswap().reshape(num_img, row * col).astype(np.float32) / 255
        with gzip.open(label_filename, 'rb') as f:
            _, _ = struct.unpack('>II', f.read(8))
            y = np.frombuffer(f.read(), np.uint8).byteswap()
        self.X = X
        self.y = y
        assert len(self.X) == len(self.y)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X, y = self.X[index], self.y[index]
        if self.transforms is not None:
            X = self.apply_transforms(X.reshape(28, 28, -1)).reshape(-1, 28 * 28)
            return X, y
        else:
            return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION