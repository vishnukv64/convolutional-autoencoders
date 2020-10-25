from torch.utils.data import Dataset
from glob import glob
from PIL import Image


class CatKingdom(Dataset):
    def __init__(self, images, transform):
        self.images_path = glob(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        images = Image.open(self.images_path[item]).convert("RGB")
        transforms = self.transform(images)

        return transforms
