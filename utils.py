from scipy import io
import os
import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset
from PIL import Image

class DatasetPair(Dataset):
    def __init__(self, cover_dir, stego_dir,
                 transform):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        # print(cover_dir)
        self.cover_list = os.listdir(cover_dir)
        self.transform = transform
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        idx = int(idx)
        labels = np.array([0,1], dtype='int32')
        cover_path = os.path.join(self.cover_dir,
                                  self.cover_list[idx])
        # print(cover_path)
        load_mat = self.cover_list[idx].endswith('.mat')
        if load_mat:
            cover = io.loadmat(cover_path)['I_spatial']
            # print(cover.size[0])#(256, 256)
            # print(type(cover))
            cover += 128
            # print(cover)
        else:
            cover = Image.open(cover_path).convert('L')
            # print("cover.shape")
            # print(cover.size)
            # print(cover)
            cover = np.array(cover)
            # print(cover.shape)
        images = np.empty((2, cover.shape[0], cover.shape[1], 1),
                          dtype='uint8')
        images[0,:,:,0] = cover
        stego_path = os.path.join(self.stego_dir,
                                      self.cover_list[idx])
        if load_mat:
            stego = io.loadmat(stego_path)['I_spatial']
            stego += 128
            # print(image)
        else:
            stego = Image.open(stego_path).convert('L')
        images[1,:,:,0] = np.array(stego)
        samples = {'images': images, 'labels': labels}
        if self.transform:
            samples = self.transform(samples)
        return samples



class DatasetPair1(Dataset):
    def __init__(self, cover_dir, stego_dir,
                 transform):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        # print(cover_dir)
        self.cover_list = os.listdir(cover_dir)
        self.transform = transform
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        idx = int(idx)
        labels = np.array([0,1], dtype='int32')
        cover_path = os.path.join(self.cover_dir,
                                  self.cover_list[idx])
        # print(cover_path)
        load_mat = self.cover_list[idx].endswith('.mat')
        if load_mat:
            cover = io.loadmat(cover_path)['I_spatial']
            # print(cover.size[0])#(256, 256)
            # print(type(cover))
            cover += 128
            # print(cover)
        else:
            cover = Image.open(cover_path)
            # print("cover.shape")
            # print(cover.size)
            # print(cover)
            cover = np.array(cover)
            # print(cover.shape)
        images = np.empty((2, cover.shape[0], cover.shape[1], 3),
                          dtype='uint8')
        images[0,:,:,:] = cover
        stego_path = os.path.join(self.stego_dir,
                                      self.cover_list[idx])
        if load_mat:
            stego = io.loadmat(stego_path)['I_spatial']
            stego += 128
            # print(image)
        else:
            stego = Image.open(stego_path)
        images[1,:,:,:] = np.array(stego)
        samples = {'images': images, 'labels': labels}
        if self.transform:
            samples = self.transform(samples)
        return samples


class DatasetMask(Dataset):
    def __init__(self, cover_dir, stego_dir,
                 transform):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        # print(cover_dir)
        self.cover_list = os.listdir(cover_dir)
        self.transform = transform
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        idx = int(idx)
        labels = np.array([0,1], dtype='int32')
        cover_path = os.path.join(self.cover_dir,
                                  self.cover_list[idx])

        load_mat = self.cover_list[idx].endswith('.mat')
        if load_mat:
            cover = io.loadmat(cover_path)['I_spatial']
            # print(cover.size[0])#(256, 256)
            # print(type(cover))
            cover += 128
            # print(cover)
        else:
            cover = Image.open(cover_path)
            cover = np.array(cover)
            # print(type(cover))
        images = np.empty((2, cover.shape[0], cover.shape[1], 1),
                          dtype='uint8')
        images[0,:,:,0] = cover
        stego_path = os.path.join(self.stego_dir,
                                      self.cover_list[idx])
        if load_mat:
            stego = io.loadmat(stego_path)['I_spatial']
            stego += 128
            # print(image)
        else:
            stego = Image.open(stego_path)
        images[1,:,:,0] = np.array(stego)
        masks = np.empty((2, cover.shape[0], cover.shape[1], 1),
                          dtype='uint8')
        masks[0, :, :, 0] = np.zeros([cover.shape[0], cover.shape[1]])
        masks[1, :, :, 0] = np.abs(images[0,:,:,0] -images[1,:,:,0])
        samples = {'images': images, 'labels': labels, 'masks':masks}
        if self.transform:
            samples = self.transform(samples)
        return samples

class ToTensor(object):
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']
        images = (images.transpose((0,3,1,2)).astype('float32') / 255)
        return {'images': torch.from_numpy(images),
                'labels': torch.from_numpy(labels).long()}

class ToTensor2(object):
    def __call__(self, samples):
        images, labels, names = samples['images'], samples['labels'], samples['names']
        images = (images.astype('float32')/ 255)
        labels = (labels.astype('float32')/ 255)
        return {'images': torch.from_numpy(images),
                'labels': torch.from_numpy(labels),
                'names' : names}

class ToTensor3(object):
    def __call__(self, samples):
        images, labels, masks = samples['images'], samples['labels'], samples['masks']
        images = (images.transpose((0, 3, 1, 2)).astype('float32')/ 255)
        masks = (masks.transpose((0, 3, 1, 2)).astype('float32')/ 255)
        return {'images': torch.from_numpy(images),
                'masks': torch.from_numpy(masks),
                'labels': torch.from_numpy(labels).long()}

class AugData():
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']

        # Rotation
        rot = random.randint(0, 3)
        images = np.rot90(images, rot, axes=[1, 2]).copy()
        # Mirroring
        if random.random() < 0.5:
            images = np.flip(images, axis=2).copy()

        new_sample = {'images': images, 'labels': labels}

        return new_sample

class AugData2():
    def __call__(self, samples):
        images, labels, names = samples['images'], samples['labels'],samples['names']

        # Rotation
        rot = random.randint(0, 3)
        images = np.rot90(images, rot, axes=[1, 2]).copy()
        labels = np.rot90(labels, rot, axes=[1, 2]).copy()
        # Mirroring
        if random.random() < 0.5:
            images = np.flip(images, axis=2).copy()
            labels = np.flip(labels, axis=2).copy()

        new_sample = {'images': images, 'labels': labels,'names': names}

        return new_sample

class AugData3():
    def __call__(self, samples):
        images, labels, masks = samples['images'], samples['labels'],samples['masks']

        # Rotation
        rot = random.randint(0, 3)
        images = np.rot90(images, rot, axes=[1, 2]).copy()
        masks = np.rot90(masks, rot, axes=[1, 2]).copy()
        # Mirroring
        if random.random() < 0.5:
            images = np.flip(images, axis=2).copy()
            masks = np.flip(masks, axis=2).copy()

        new_sample = {'images': images, 'labels': labels,'masks': masks}

        return new_sample

class DatasetDenoise(Dataset):
    def __init__(self, cover_dir, stego_dir,
                 transform):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        # print(cover_dir)
        self.cover_list = os.listdir(cover_dir)
        self.transform = transform
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        idx = int(idx)
        cover_path = os.path.join(self.cover_dir,
                                  self.cover_list[idx])
        cover = Image.open(cover_path)
        cover = np.array(cover)
            # print(type(cover))
        images = np.empty((1,cover.shape[0], cover.shape[1]),
                          dtype='uint8')
        labels = np.empty((1, cover.shape[0], cover.shape[1]),
                          dtype='uint8')
        stego_path = os.path.join(self.stego_dir,
                                      self.cover_list[idx])

        stego = np.array(Image.open(stego_path))
        images[0,:,:] = stego
        labels[0,:,:] = np.abs(cover - stego)
        samples = {'images': images, 'labels': labels,'names':self.cover_list[idx]}
        if self.transform:
            samples = self.transform(samples)
        return samples