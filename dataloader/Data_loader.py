import shutil
import pandas as pd
from utils.toolfuc import *
import numpy as np
from torch.utils.data import Dataset
import h5py
import torch
import imgaug.augmenters as iaa


def collect_serial_dir_file(serial_path, match='.h5'):
    files_list = []
    for root, dir, files in os.walk(serial_path):
        for f in files:
            if match.lower() in f.lower():
                files_list.append(os.path.join(root, f))
    return files_list


def get_data_path(path, excelpath):  # 把所有h5文件的path都汇总到path_list
    excel = pd.read_csv(excelpath, header=0)
    serial_list = excel['serial']
    path_list = []
    for i in range(len(serial_list)):
        h5path = os.path.join(path, str(serial_list[i]))
        data_path = collect_serial_dir_file(h5path)  # 同文件里有一个固定随机种子的函数
        for f in data_path:
            path_list.append(f)
    return path_list


class DatasetGenerator(Dataset):
    def __init__(self, path, excelpath, Aug=False, n_class=2, set_name='train'):
        self.Aug = Aug
        self.n_class = n_class
        self.set_name = set_name
        if set_name == 'train':
            self.data_path = get_data_path(path, excelpath)
        if set_name == 'vaild':
            self.data_path = get_data_path(path, excelpath)
        if set_name == 'test':
            self.data_path = get_data_path(path, excelpath)

        sometimes = lambda aug: iaa.Sometimes(0.9, aug)  # 设定随机函数,90%几率扩增,or
        self.seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # 50%图像进行水平翻转
                iaa.Flipud(0.5),  # 50%图像做垂直翻转
                sometimes(iaa.Crop(percent=(0, 0.1))),  # 对随机的一部分图像做crop操作 crop的幅度为0到10%
                sometimes(iaa.Affine(  # 对一部分图像做仿射变换
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # 图像缩放为80%到120%之间
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # 平移±20%之间
                    rotate=(-20, 20),  # 旋转±20度之间
                    shear=(-10, 10),  # 剪切变换±16度，（矩形变平行四边形）
                    order=[0, 1],  # 使用最邻近差值或者双线性差值
                    # mode=ia.ALL,  # 边缘填充
                )),
            ],
            random_order=True  # 随机的顺序把这些操作用在图像上
        )

    def __getitem__(self, index):
        data_path = self.data_path[index]
        # copy to local disk
        nas3_rootdir = '/home/user14/sharedata/ZChang'
        if not os.path.exists(data_path.replace('/home/user14/sharedata/newnas/ZChang', nas3_rootdir)):
            # get folder path
            folder_path = os.path.dirname(data_path)
            folder_path = folder_path.replace('/home/user14/sharedata/newnas/ZChang', nas3_rootdir)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # copy file
            shutil.copy(data_path, folder_path)
            # open file
        data = h5py.File(data_path.replace('/home/user14/sharedata/newnas/ZChang', nas3_rootdir), mode='r')

        image = np.squeeze(data['Data'][()])
        mask = np.squeeze(data['Mask'][()])
        label = data['Label'][()]
        sign = int(data_path.split("/")[-1].split(".")[0].split("_")[0])

        if self.Aug is True:

            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=3)
            mask = np.expand_dims(mask, axis=0)
            mask = np.expand_dims(mask, axis=3).astype(np.int32)

            imager, masker = self.seq(images=image, segmentation_maps=mask)
            imageg, imageb = imager, imager
            img_rg = np.concatenate([imager, imageg], axis=0)
            img_rgb = np.concatenate([img_rg, imageb], axis=0)
            img_rgb = img_rgb[:, :, :, 0]
            mask_rgb = masker[0, :, :, 0]

        else:
            imager = np.expand_dims(image, axis=0)
            masker = mask.astype(np.int32)
            imageg, imageb = imager, imager
            img_rg = np.concatenate([imager, imageg], axis=0)
            img_rgb = np.concatenate([img_rg, imageb], axis=0)
            mask_rgb = masker

        img_rgb = np.asarray(img_rgb).astype('float32')
        mask_rgb = np.asarray(mask_rgb).astype('float32')

        return torch.FloatTensor(img_rgb), torch.FloatTensor(mask_rgb), torch.FloatTensor(label), sign

    def __len__(self):
        return len(self.data_path)
