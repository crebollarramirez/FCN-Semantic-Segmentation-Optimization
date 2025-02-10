import os
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms

num_classes = 21
ignore_label = 255
root = './data'

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''


palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]  #3 values- R,G,B for every class. First 3 values for class 0, next 3 for


def make_dataset(mode):
    """
    Creates a list of tuples (image_path, mask_path) for a given dataset mode.

    Asserts that the mode is one of 'train', 'val', or 'test'.
    Based on the mode, it reads corresponding image and mask paths from the VOC dataset.
    For each image, it pairs its path with the corresponding mask path.

    data_list for train is with name train.txt,
    data_list for validation and test is with name val.txt.

    Args:
        mode (str): The mode of the dataset, either 'train', 'valtest'.

    Returns:
        list of tuples: Each tuple contains paths (image_path, mask_path).
    """
    assert mode in ['train', 'val']
    items = []

    # FOR TRAIN SET
    img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
    data_list = [l.strip('\n') for l in open(os.path.join(
        root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', ('train.txt' if mode == "train" else "val.txt"))).readlines()]
    for it in data_list:
        item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
        items.append(item)

    return items



class VOC(data.Dataset):
    """
    A custom dataset class for VOC dataset.

    - Resizes images and masks to a specified width and height.
    - Implements methods to get dataset items and dataset length.

    Args:
        mode (str): Mode of the dataset ('train', 'val', etc.).
        transform (callable, optional): Transform to be applied to the images.
        target_transform (callable, optional): Transform to be applied to the masks.
    """

    def __init__(self, mode, transform=None, target_transform=None, modification=False):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.width = 224
        self.height = 224
        self.modification = modification

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB').resize((self.width, self.height), Image.BILINEAR)
        mask = Image.open(mask_path).resize((self.width, self.height), Image.NEAREST)

        # Apply the same transformations to both img and mask
        if self.modification:
            augmentation_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomRotation(5),
                transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.0))
            ])
            seed = random.randint(0, 2**32)
            random.seed(seed)
            img = augmentation_transforms(img)
            random.seed(seed)
            mask = augmentation_transforms(mask)

        # normalization for image
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        mask[mask==ignore_label]=0

        return img, mask

    def __len__(self):
        return len(self.imgs)
