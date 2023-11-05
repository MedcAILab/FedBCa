import os

import h5py
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

global nii_name
import pandas as pd
import cv2


def nii_loader(nii_path):
    print('#Loading ', nii_path, '...')
    data = nib.load(nii_path)

    print("--Loading size:", data.shape)

    return data


def my_resize(o_data, transform_size=None, transform_rate=None):
    print('#Resizing...')
    data = o_data
    print("--Original size:", data.shape)
    if transform_size:
        o_width, o_height, o_queue = data.shape
        width, height, queue = transform_size
        data = zoom(data, (width / o_width, height / o_height, queue / o_queue))
    elif transform_rate:
        data = zoom(data, transform_rate)

    print("--Transofmed size:", data.shape)

    return data


def centre_window_cropping(o_data, reshapesize=None):
    print('#Centre window cropping...')
    data = o_data
    or_size = data.shape
    target_size = (reshapesize[0], reshapesize[1], or_size[2])

    # pad if or_size is smaller than target_size
    if (target_size[0] > or_size[0]) | (target_size[1] > or_size[1]):
        if target_size[0] > or_size[0]:
            pad_size = int((target_size[0] - or_size[0]) / 2)
            data = np.pad(data, ((pad_size, pad_size), (0, 0), (0, 0)))
        if target_size[1] > or_size[1]:
            pad_size = int((target_size[1] - or_size[1]) / 2)
            data = np.pad(data, ((0, 0), (pad_size, pad_size), (0, 0)))

    #  centre_window_cropping
    cur_size = data.shape
    centre_x = float(cur_size[0] / 2)
    centre_y = float(cur_size[1] / 2)
    dx = float(target_size[0] / 2)
    dy = float(target_size[1] / 2)
    data = data[int(centre_x - dx + 1):int(centre_x + dx), int(centre_y - dy + 1): int(centre_y + dy), :]

    data_resize = np.zeros((reshapesize[0], reshapesize[1], cur_size[2]))
    for kk in range(cur_size[2]):
        data_resize[:, :, kk] = cv2.resize(data[:, :, kk], (reshapesize[0], reshapesize[1]),
                                           interpolation=cv2.INTER_NEAREST)

    return data_resize


def getListIndex(arr, value):
    dim1_list = dim2_list = dim3_list = []
    if (arr.ndim == 3):
        index = np.argwhere(arr == value)
        dim1_list = index[:, 0].tolist()
        dim2_list = index[:, 1].tolist()
        dim3_list = index[:, 2].tolist()

    else:
        raise ValueError('The ndim of array must be 3!!')

    return dim1_list, dim2_list, dim3_list


def ROI_cutting(o_data, o_roi, expend_voxel=1):
    print('#ROI cutting...')
    data = o_data
    roi = o_roi

    [I1, I2, I3] = getListIndex(roi, 1)
    d1_min = min(I1)
    d1_max = max(I1)
    d2_min = min(I2)
    d2_max = max(I2)
    d3_min = min(I3)
    d3_max = max(I3)
    print(d3_min, d3_max)

    if expend_voxel > 0:
        d1_min -= expend_voxel
        d1_max += expend_voxel
        d2_min -= expend_voxel
        d2_max += expend_voxel
        d3_min -= 1
        d3_max += 1

        d1_min = d1_min if d1_min > 0 else 0
        d1_max = d1_max if d1_max < data.shape[0] - 1 else data.shape[0] - 1
        d2_min = d2_min if d2_min > 0 else 0
        d2_max = d2_max if d2_max < data.shape[1] - 1 else data.shape[1] - 1
        d3_min = d3_min if d3_min > 0 else 0
        d3_max = d3_max if d3_max < data.shape[2] - 1 else data.shape[2] - 1

    data = data[d1_min:d1_max, d2_min:d2_max, d3_min:d3_max]
    print(data.shape)
    roi = roi[d1_min:d1_max, d2_min:d2_max, d3_min:d3_max]

    print("--Cutting size:", data.shape)
    return data, roi


def make_h5_data(o_data, o_roi, label=None, h5_save_path=None, count=None, count1=None):
    print('#Make h5 data...')
    data = o_data
    roi = o_roi
    if (h5_save_path):
        for i, (divided_data, divided_roi) in enumerate(zip(data, roi)):
            if not os.path.exists(os.path.join(h5_save_path, str(count))):
                os.makedirs(os.path.join(h5_save_path, str(count)))
            save_file_name = os.path.join(h5_save_path, str(count), str(count) + '_' + str(i + 1) + '.h5')
            with h5py.File(save_file_name, 'a') as f:
                print("--h5 file path:", save_file_name, '    -label:', label, '    -size:', divided_data.shape)
                f['Data'] = divided_data
                f['Label'] = [label]
                f['Mask'] = divided_roi


def make_h5_data_new(o_data, o_roi=None, label=None, h5_save_path=None, count=None, count1=None):
    print('#Make h5 data...')
    data = o_data
    roi = o_roi
    if (h5_save_path):
        if not os.path.exists(os.path.join(h5_save_path, str(count))):
            os.makedirs(os.path.join(h5_save_path, str(count)))
    save_file_name = os.path.join(h5_save_path, str(count), str(count) + '_' + str(count) + '.h5')
    with h5py.File(save_file_name, 'a') as f:
        print("--h5 file path:", save_file_name, '    -label:', label, '    -size:', data.shape)
        f['Data'] = data
        f['Label'] = [label]
        f['ROI'] = roi
        print('kokokokokokok')


def linear_normalizing(o_data):
    print('#Linear_normalizing...')
    data = o_data
    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min)

    return data


def block_dividing(o_data, deep=None, step=None):
    print('#Block dividing...')
    data = o_data
    data_group = []
    o_data_deep = data.shape[2]

    if o_data_deep <= deep:
        tmp_data = np.zeros((data.shape[0], data.shape[1], deep))
        tmp_data[:, :, 0:o_data_deep] = data
        blocks = 1
        tmp_data = tmp_data
        data_group.append(tmp_data)

    else:
        blocks = (o_data_deep - deep) // step + 2
        if (o_data_deep - deep) % step == 0:
            blocks -= 1
        for i in range(blocks - 1):
            tmp_data = data[:, :, (0 + i * step): (deep + i * step)]
            data_group.append(tmp_data)
        tmp_data = data[:, :, o_data_deep - deep:o_data_deep]
        data_group.append(tmp_data)

    print("--Block size:", tmp_data.shape,
          " Divided number:(%d)" % (blocks))

    return data_group, blocks


if __name__ == "__main__":

    Center1_preprocessing = True
    Center2_preprocessing = False
    Center3_preprocessing = False
    Center4_preprocessing = False

    """ basic parameters """
    reshapesize = (128, 128)
    deep = 1
    step = 1
    expend_voxel = 5
    Center1_count = 0
    Center2_count = 0
    Center3_count = 0
    Center4_count = 0
    """Center_1 path """
    Center1_roi_root = r'D:\Bca-data\Center_1\Annotation'
    Center1_data_root = r'D:\Bca-data\Center_1\T2WI'
    Center1_roi_list = os.listdir(Center1_roi_root)
    Center1_save_root = r"D:\Bca-data\process_classification\Center_1"
    if not os.path.exists(Center1_save_root):
        os.makedirs(Center1_save_root)

    """Center_2 path """
    Center2_roi_root = r'D:\Bca-data\Center_2\Annotation'
    Center2_data_root = r'D:\Bca-data\Center_2\T2WI'
    Center2_save_root = r"D:\Bca-data\process_classification\Center_2"
    if not os.path.exists(Center2_save_root):
        os.makedirs(Center2_save_root)
    Center2_roi_list = os.listdir(Center2_roi_root)

    """Center_3 path """
    Center3_roi_root = r'D:\Bca-data\Center_3\Annotation'
    Center3_data_root = r'D:\Bca-data\Center_3\T2WI'
    Center3_save_root = r"D:\Bca-data\process_classification\Center_3"
    if not os.path.exists(Center3_save_root):
        os.makedirs(Center3_save_root)
    Center3_roi_list = os.listdir(Center3_roi_root)

    """Center_4 path """
    Center4_roi_root = r'D:\Bca-data\Center_4\Annotation'
    Center4_data_root = r'D:\Bca-data\Center_4\T2WI'
    Center4_save_root = r"D:\Bca-data\process_classification\Center_4"
    if not os.path.exists(Center4_save_root):
        os.makedirs(Center4_save_root)
    Center4_roi_list = os.listdir(Center4_roi_root)

    """ makedirs """
    Center1_h5_save_path = os.path.join(Center1_save_root, 'h5_data_128')
    if not os.path.exists(Center1_h5_save_path):
        os.makedirs(Center1_h5_save_path)

    Center2_h5_save_path = os.path.join(Center2_save_root, 'h5_data_128')
    if not os.path.exists(Center2_h5_save_path):
        os.makedirs(Center2_h5_save_path)

    Center3_h5_save_path = os.path.join(Center3_save_root, 'h5_data_128')
    if not os.path.exists(Center3_h5_save_path):
        os.makedirs(Center3_h5_save_path)

    Center4_h5_save_path = os.path.join(Center4_save_root, 'h5_data_128')
    if not os.path.exists(Center4_h5_save_path):
        os.makedirs(Center4_h5_save_path)

    """Reading the label information"""
    Center1_label_list = pd.read_excel(r'D:\Bca-data\Center_1\Center1_label.xlsx')
    Center1_label_list = np.array(Center1_label_list)

    Center2_label_list = pd.read_excel(
        r'D:\Bca-data\Center_2\Center2_label.xlsx')
    Center2_label_list = np.array(Center2_label_list)

    Center3_label_list = pd.read_excel(
        r'D:\Bca-data\Center_3\Center3_label.xlsx')
    Center3_label_list = np.array(Center3_label_list)

    Center4_label_list = pd.read_excel(
        r'D:\Bca-data\Center_4\Center4_label.xlsx')
    Center4_label_list = np.array(Center4_label_list)

    """Readint the image path and list"""
    Center1_all_information = []
    Center2_all_information = []
    Center3_all_information = []
    Center4_all_information = []

    """Reading the roi path and list"""
    Center1_roi_list = os.listdir(Center1_roi_root)
    Center1_roi_list.sort()

    Center2_roi_list = os.listdir(Center2_roi_root)
    Center2_roi_list.sort()

    Center3_roi_list = os.listdir(Center3_roi_root)
    Center3_roi_list.sort()

    Center4_roi_list = os.listdir(Center4_roi_root)
    Center4_roi_list.sort()

    """Processing the data"""
    if Center1_preprocessing:
        for filename in Center1_roi_list:
            Center1_count += 1
            case_name = filename.split(".")[0].split("_")[0]
            print("case_name:{}".format(case_name))
            roi_data_path = os.path.join(Center1_roi_root, filename)
            img_data_path = os.path.join(Center1_data_root, case_name + ".nii.gz")

            data_metrix = nii_loader(img_data_path)
            roi_metrix = nii_loader(roi_data_path)

            Center1_label = Center1_label_list[Center1_label_list[:, 2] == filename, 0]
            print("label:{}".format(Center1_label))
            roi_arr = np.array(roi_metrix.dataobj, dtype='float32')
            roi_arr[roi_arr < 0.5] = 0
            roi_arr[roi_arr >= 0.5] = 1
            img_arr = np.array(data_metrix.dataobj, dtype='float32')

            img_arr, roi_arr = ROI_cutting(img_arr, roi_arr, expend_voxel=expend_voxel)
            img_arr = centre_window_cropping(img_arr, reshapesize=reshapesize)
            roi_arr = centre_window_cropping(roi_arr, reshapesize=reshapesize)
            img_arr = linear_normalizing(img_arr)
            img_arr = block_dividing(img_arr, deep=deep, step=step)
            roi_arr = block_dividing(roi_arr, deep=deep, step=step)
            make_h5_data(img_arr[0], roi_arr[0], label=Center1_label[0], h5_save_path=Center1_h5_save_path,
                         count=Center1_count)
            Center1_all_information.append([Center1_count, int(Center1_label), filename])

        print("all_information:{}".format(Center1_all_information))
        print("all_information_shape:{}".format(np.shape(Center1_all_information)))
        Center1_all_information = np.array(Center1_all_information)
        Center1_all_information_pd = pd.DataFrame(
            {"serial": Center1_all_information[:, 0], "label": Center1_all_information[:, 1],
             "filename": Center1_all_information[:, 2]})
        Center1_all_information_pd.to_excel("D:\Bca-data\process_classification\Center_1\h5_information.xlsx")

    if Center2_preprocessing:
        for filename in Center2_roi_list:
            Center2_count += 1
            case_name = filename.split(".")[0].split("_")[0]
            print("case_name:{}".format(case_name))
            roi_data_path = os.path.join(Center2_roi_root, filename)
            img_data_path = os.path.join(Center2_data_root, case_name + ".nii.gz")

            data_metrix = nii_loader(img_data_path)
            roi_metrix = nii_loader(roi_data_path)

            Center2_label = Center2_label_list[Center2_label_list[:, 2] == filename, 0]
            print("label:{}".format(Center2_label))
            roi_arr = np.array(roi_metrix.dataobj, dtype='float32')
            roi_arr[roi_arr < 0.5] = 0
            roi_arr[roi_arr >= 0.5] = 1
            img_arr = np.array(data_metrix.dataobj, dtype='float32')

            img_arr, roi_arr = ROI_cutting(img_arr, roi_arr, expend_voxel=expend_voxel)
            img_arr = centre_window_cropping(img_arr, reshapesize=reshapesize)
            roi_arr = centre_window_cropping(roi_arr, reshapesize=reshapesize)
            img_arr = linear_normalizing(img_arr)
            img_arr = block_dividing(img_arr, deep=deep, step=step)
            roi_arr = block_dividing(roi_arr, deep=deep, step=step)
            make_h5_data(img_arr[0], roi_arr[0], label=Center2_label[0], h5_save_path=Center2_h5_save_path,
                         count=Center2_count)
            Center2_all_information.append([Center2_count, int(Center2_label), filename])

        print("all_information:{}".format(Center2_all_information))
        print("all_information_shape:{}".format(np.shape(Center2_all_information)))
        Center2_all_information = np.array(Center2_all_information)
        Center2_all_information_pd = pd.DataFrame(
            {"serial": Center2_all_information[:, 0], "label": Center2_all_information[:, 1],
             "filename": Center2_all_information[:, 2]})
        Center2_all_information_pd.to_excel(
            "D:\Bca-data\process_classification\Center_2\h5_information.xlsx")

    if Center3_preprocessing:
        for filename in Center3_roi_list:
            Center3_count += 1
            case_name = filename.split(".")[0].split("_")[0]
            print("case_name:{}".format(case_name))
            roi_data_path = os.path.join(Center3_roi_root, filename)
            img_data_path = os.path.join(Center3_data_root, case_name + ".nii.gz")

            data_metrix = nii_loader(img_data_path)
            roi_metrix = nii_loader(roi_data_path)

            Center3_label = Center3_label_list[Center3_label_list[:, 2] == filename, 0]
            print("label:{}".format(Center3_label))
            roi_arr = np.array(roi_metrix.dataobj, dtype='float32')
            roi_arr[roi_arr < 0.5] = 0
            roi_arr[roi_arr >= 0.5] = 1
            img_arr = np.array(data_metrix.dataobj, dtype='float32')

            img_arr, roi_arr = ROI_cutting(img_arr, roi_arr, expend_voxel=expend_voxel)
            img_arr = centre_window_cropping(img_arr, reshapesize=reshapesize)
            roi_arr = centre_window_cropping(roi_arr, reshapesize=reshapesize)
            img_arr = linear_normalizing(img_arr)
            img_arr = block_dividing(img_arr, deep=deep, step=step)
            roi_arr = block_dividing(roi_arr, deep=deep, step=step)
            make_h5_data(img_arr[0], roi_arr[0], label=Center3_label[0], h5_save_path=Center3_h5_save_path,
                         count=Center3_count)
            Center3_all_information.append([Center3_count, int(Center3_label), filename])

        print("all_information:{}".format(Center3_all_information))
        print("all_information_shape:{}".format(np.shape(Center3_all_information)))
        Center3_all_information = np.array(Center3_all_information)
        Center3_all_information_pd = pd.DataFrame(
            {"serial": Center3_all_information[:, 0], "label": Center3_all_information[:, 1],
             "filename": Center3_all_information[:, 2]})
        Center3_all_information_pd.to_excel(
            "D:\Bca-data\process_classification\Center_3\h5_information.xlsx")

    if Center4_preprocessing:
        for filename in Center4_roi_list:
            Center4_count += 1
            case_name = filename.split(".")[0].split("_")[0]
            print("case_name:{}".format(case_name))
            roi_data_path = os.path.join(Center4_roi_root, filename)
            img_data_path = os.path.join(Center4_data_root, case_name + ".nii.gz")

            data_metrix = nii_loader(img_data_path)
            roi_metrix = nii_loader(roi_data_path)

            Center4_label = Center4_label_list[Center4_label_list[:, 2] == filename, 0]
            print("label:{}".format(Center4_label))
            roi_arr = np.array(roi_metrix.dataobj, dtype='float32')
            roi_arr[roi_arr < 0.5] = 0
            roi_arr[roi_arr >= 0.5] = 1
            img_arr = np.array(data_metrix.dataobj, dtype='float32')

            img_arr, roi_arr = ROI_cutting(img_arr, roi_arr, expend_voxel=expend_voxel)
            img_arr = centre_window_cropping(img_arr, reshapesize=reshapesize)
            roi_arr = centre_window_cropping(roi_arr, reshapesize=reshapesize)
            img_arr = linear_normalizing(img_arr)
            img_arr = block_dividing(img_arr, deep=deep, step=step)
            roi_arr = block_dividing(roi_arr, deep=deep, step=step)
            make_h5_data(img_arr[0], roi_arr[0], label=Center4_label[0], h5_save_path=Center4_h5_save_path,
                         count=Center4_count)
            Center4_all_information.append([Center4_count, int(Center4_label), filename])

        print("all_information:{}".format(Center4_all_information))
        print("all_information_shape:{}".format(np.shape(Center4_all_information)))
        Center4_all_information = np.array(Center4_all_information)
        Center4_all_information_pd = pd.DataFrame(
            {"serial": Center4_all_information[:, 0], "label": Center4_all_information[:, 1],
             "filename": Center4_all_information[:, 2]})
        Center4_all_information_pd.to_excel(
            "D:\Bca-data\process_classification\Center_4\h5_information.xlsx")
