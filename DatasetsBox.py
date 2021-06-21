import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
import random
import math
import cv2
from skimage import transform as trans
import numbers
from torchvision.transforms import functional as F

def point_rotate(point, row, theta, center):
    point = np.asarray([[point[0], point[1]], [point[2], point[3]]])
    x1 = point[:, 0]
    y1 = row - point[:, 1]
    x2 = center[0]
    y2 = row - center[1]
    x = (x1 - x2) * math.cos(math.pi / 180.0 * theta) - (y1 - y2) * math.sin(math.pi / 180.0 * theta) + x2
    y = (x1 - x2) * math.sin(math.pi / 180.0 * theta) + (y1 - y2) * math.cos(math.pi / 180.0 * theta) + y2
    y = row - y
    point = np.vstack((x, y)).T
    point = [point[0][0], point[0][1], point[1][0], point[1][1]]
    return point


class RandomRotate(object):
    def __init__(self, degree):
        assert isinstance(degree, (int, tuple))
        self.degrees = (-degree, degree)

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        img, box = sample['image'], sample['box']
        cent = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        angle = self.get_params(self.degrees)
        img = img.rotate(angle, center=cent)
        row = img.size[1]
        box_out = point_rotate(box, row, angle, cent)

        return {'image': img, 'box': box_out, 'label': sample['label']}


class RandomTrans(object):
    def __init__(self, trans_rate):
        assert isinstance(trans_rate, float)
        self.trans_rate = trans_rate

    def __call__(self, sample):
        img, box, label = sample['image'], sample['box'], sample['label']
        w = box[2] - box[0]
        h = box[3] - box[1]
        transize = np.random.random(4) * 2 * self.trans_rate - self.trans_rate
        box[0] += transize[0] * w
        box[2] += transize[2] * w
        box[1] += transize[1] * h
        box[3] += transize[3] * h

        return {'image': img, 'box': box, 'label': label}


def make_square(bboxA):
    o_h = bboxA[3] - bboxA[1]
    bboxA[3] = bboxA[3] + o_h / 6
    # bboxA[1] = bboxA[1] + o_h / 8

    h = bboxA[3] - bboxA[1]
    w = bboxA[2] - bboxA[0]
    l = np.maximum(w, h)
    bboxA[0] = bboxA[0] + w * 0.5 - l * 0.5
    bboxA[1] = bboxA[1] + h * 0.5 - l * 0.5
    bboxA[2:4] = bboxA[0:2] + l
    return bboxA


def align_face(img, bbox):
    facesize = int(bbox[2] - bbox[0])
    image_size = (facesize, facesize)
    src = np.array([
        [0, 0],
        [0, facesize],
        [facesize, 0],
        [facesize, facesize]], dtype=np.float32)
    point = np.array([
        [bbox[0], bbox[1]],
        [bbox[0], bbox[3]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]]], dtype=np.float32)
    dst = point.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    imgcv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    image = cv2.warpAffine(imgcv, M, (image_size[1], image_size[0]), borderValue=0.0)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


class CropResize(object):
    def __init__(self, output_size, train=True):
        assert isinstance(output_size, int)
        self.size = (output_size, output_size)
        self.train = train

    def __call__(self, sample):
        img, box, label = sample['image'], sample['box'], sample['label']
        box = make_square(box)
        face = align_face(img, box)
        face = F.resize(face, self.size)

        return {'image': face, 'box': box, 'label': label}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        img, box, label = sample['image'], sample['box'], sample['label']
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        img = transform(img)
        return {'image': img, 'box': box, 'label': label}

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, box, label = sample['image'], sample['box'], sample['label']
        if random.random() < self.p:
            img = F.hflip(img)

        return {'image': img, 'box': box, 'label': label}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ToTensor(object):
    def __call__(self, sample):
        rgbimg, box, label = sample['image'], sample['box'], sample['label']
        hsvimg = rgbimg.convert('HSV')
        rgbimg = F.to_tensor(rgbimg)
        hsvimg = F.to_tensor(hsvimg)
        catimg = torch.cat([rgbimg, hsvimg], 0)
        return {'image': catimg, 'box': box, 'label': label}

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        tensor, box, label = sample['image'], sample['box'], sample['label']
        tensor = F.normalize(tensor, self.mean, self.std)
        return {'image': tensor, 'box': box, 'label': label}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class DatasetLoaderTrain(Dataset):
    def __init__(self, imgroot, name, getreal, transform=None):
        labelroot = os.path.join('./csv', name)
        if getreal:
            filename = os.path.join(labelroot, 'image_list_real.csv')
        else:
            filename = os.path.join(labelroot, 'image_list_fake.csv')

        self.train_frame = pd.read_csv(filename)
        self.transform = transform
        self.imgroot = imgroot

    def __getitem__(self, idx):
        img_path = self.train_frame.iloc[idx, 0]
        img_path = os.path.join(self.imgroot, img_path)
#         print(img_path)
        image = Image.open(img_path).convert('RGB')
        box = self.train_frame.iloc[idx, 1]
        box = eval(box)
#         print(box)
        label = int(self.train_frame.iloc[idx, 2])
        sample = {'image': image, 'box': box, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], label

    def __len__(self):
        return len(self.train_frame)


def train_dataset_loader(imgroot, name, getreal, batch_size, trainlevel=0):
    net_input = 256
    if trainlevel == 0:
        pre_process = transforms.Compose([RandomRotate(10), RandomTrans(0.1),
                                          CropResize(net_input),
                                          ColorJitter(0.3, 0.3, 0.3, 0.2),
                                          RandomHorizontalFlip(),
                                          ToTensor(),
                                          Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])
    if trainlevel == 1:
        pre_process = transforms.Compose([RandomRotate(15), RandomTrans(0.15),
                                          CropResize(net_input),
                                          ColorJitter(0.4, 0.4, 0.4, 0.3),
                                          RandomHorizontalFlip(),
                                          ToTensor(),
                                          Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])
    if trainlevel == 2:
        pre_process = transforms.Compose([RandomRotate(20), RandomTrans(0.20),
                                          CropResize(net_input),
                                          ColorJitter(0.5, 0.5, 0.5, 0.5),
                                          RandomHorizontalFlip(),
                                          ToTensor(),
                                          Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])


    # dataset and data loader
    dataset = DatasetLoaderTrain(imgroot=imgroot, name=name, getreal=getreal, transform=pre_process)
    kwargs = {'num_workers': 4, 'pin_memory': True, 'batch_size': batch_size,
              'shuffle': True, 'drop_last': True}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return data_loader


class DatasetLoaderTest(Dataset):
    def __init__(self, imgroot, name, transform=None):
        labelroot = os.path.join('./csv', name)
        filename = os.path.join(labelroot, 'image_list_all.csv')

        self.train_frame = pd.read_csv(filename)
        self.transform = transform
        self.imgroot = imgroot


    def __getitem__(self, idx):
        img_path = self.train_frame.iloc[idx, 0]
        img_path = os.path.join(self.imgroot, img_path)
        image = Image.open(img_path).convert('RGB')
        box = self.train_frame.iloc[idx, 1]
        box = eval(box)
        sample = {'image': image, 'box': box, 'label': 0}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], img_path

    def __len__(self):
        return len(self.train_frame)


def test_dataset_loader(imgroot,name, batch_size):
    net_input = 256
    pre_process = transforms.Compose([CropResize(net_input, train=False),
                                      ToTensor(),
                                      Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])

    dataset = DatasetLoaderTest(imgroot=imgroot, name=name, transform=pre_process)
    kwargs = {'num_workers': 4, 'pin_memory': True, 'batch_size': batch_size,
              'shuffle': False, 'drop_last': False}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return data_loader


if __name__ == '__main__':
    dl = train_dataset_loader('', '1', True, 10)
    for image,label in dl:
        print(image.size())
