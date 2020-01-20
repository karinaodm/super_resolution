import os
import cv2
import torch
import torchvision
import numpy as np


class ImageDataset(torch.utils.data.Dataset):
    """Images dataset."""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        extentions = ('.jpg', '.jpeg', '.bmp', '.png')
        self.img_list = [f for f in os.listdir(root_dir) if f.endswith(extentions) and os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.img_list[idx])
        hr = cv2.imread(img_name)[...,::-1]   # convert bgr to rgb

        h, w = hr.shape[:2]
        h_lr, w_lr = int(0.5 * h), int(0.5 * w)
        lr = cv2.resize(hr, dsize=(w_lr, h_lr))
        
        sample = {'low_resolution': lr, 'high_resolution': hr}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        lr, hr = sample['low_resolution'], sample['high_resolution']

        h, w = hr.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        hr = hr[top: top + new_h,
                left: left + new_w]

        h_lr, w_lr = int(0.5 * new_h), int(0.5 * new_w)
        lr = cv2.resize(hr, dsize=(w_lr, h_lr))

        return {'low_resolution': lr, 'high_resolution': hr}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        lr, hr = sample['low_resolution'].transpose((2, 0, 1)), sample['high_resolution'].transpose((2, 0, 1))

        return {'low_resolution': torch.Tensor(lr.copy()),
                'high_resolution': torch.Tensor(hr.copy())}


def create_dataloader(data_settings, mode):
    # Create the dataset
    dataset = ImageDataset(root_dir=os.path.join(data_settings['path2data'], mode),
                           transform=torchvision.transforms.Compose([
                               RandomCrop((data_settings['height'], data_settings['width'])),
                               ToTensor(),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_settings['batch_size'],
                                             shuffle=True, num_workers=data_settings['workers'])
    return dataloader