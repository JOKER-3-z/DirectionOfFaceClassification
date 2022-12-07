import sys
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils, conv_utils


def eachFile(filepath):  # 将目录内的文件名放入列表中
    pathDir = os.listdir(filepath)
    print(pathDir)
    out = []
    for allDir in pathDir:
        out.append(allDir)
    return out


# 将图像分为训练集和测试集
def get_data(data_name, train_percentage=0.7, resize=True, data_format='channels_last'):  # 从文件夹中获取图像数据
    file_name = os.path.join(pic_dir_out, data_name + str(Width) + "X" + str(Height) + ".pkl")
    if os.path.exists(file_name):  # 判断之前是否有存到文件中
        print('INFO---------pickle already had!---------\n')
        (X_train, y_train), (X_test, y_test) = pickle.load(open(file_name, "rb"))
        return (X_train, y_train), (X_test, y_test)
        # data_format = conv_utils.normalize_data_format(data_format)
    pic_dir_set = eachFile(pic_dir_data)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label = 0
    for pic_dir in pic_dir_set:
        if not os.path.isdir(os.path.join(pic_dir_data, pic_dir)):
            continue
        pic_set = eachFile(os.path.join(pic_dir_data, pic_dir))
        pic_index = 0
        train_count = int(len(pic_set) * train_percentage)
        for pic_name in pic_set:
            if not os.path.isfile(os.path.join(pic_dir_data, pic_dir, pic_name)):
                continue
            img = cv2.imread(os.path.join(pic_dir_data, pic_dir, pic_name))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if (resize):
                img = cv2.resize(img, (Width, Height))
            if (data_format == 'channels_last'):
                img = img.reshape(-1, Width, Height, 1)
            elif (data_format == 'channels_first'):
                img = img.reshape(-1, 1, Width, Height)
            if (pic_index < train_count):
                X_train.append(img)
                y_train.append(label)
            else:
                X_test.append(img)
                y_test.append(label)
            pic_index += 1
        if len(pic_set) != 0:
            label += 1
    X_train = np.concatenate(X_train, axis=0)
    # X_test = np.concatenate(X_test,axis=0)
    y_train = np.array(y_train)
    # y_test = np.array(y_test)
    pickle.dump([(X_train, y_train), (X_test, y_test)], open(file_name, "wb"))
    return (X_train, y_train), (X_test, y_test)


def main():
    # 将增强后的图像数据转变为(X_train, y_train), (X_test, y_test)矩阵数据
    # 转化后的图像宽高和图像格式及种类等自己可以定义
    global Width, Height, num_classes, pic_dir_out, pic_dir_data
    Width = 48
    Height = 32
    num_classes = 21
    pic_dir_out = '/content/'
    pic_dir_data = '/content/CH_CAS_PEAL_POSE/'
    return get_data('CAS_IMAGES', train_percentage=1)


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = main()
    for img in X_train:
        plt.figure()
        img = img.reshape(32, 48)
        plt.imshow(img)
        plt.show()
