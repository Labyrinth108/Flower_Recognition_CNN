#coding:utf-8

import os
import PIL
from PIL import Image
import numpy as np
import shutil


def load_data(path, size, is_train_flag):
    imgs = os.listdir(path)
    num = len(imgs)
    map_dict = {"n04971313":0, "n11950345":1, "n11978233":2, "n12306717":3, "n12649065":4, "n12421683":5}

    label_num = {"n04971313":0, "n11950345":0, "n11978233":0, "n12306717":0, "n12649065":0, "n12421683":0}
    if is_train_flag:
        max_num = 540
    else:
        max_num = 130

    data = np.empty((max_num * 6, size, size, 3), dtype="float32")
    label = np.empty((max_num * 6,), dtype="uint8")
    j = 0

    for i in range(num):
        try:
            type = imgs[i].split('_')[0]
            if label_num[type] >= max_num:
                continue

            img = Image.open(path + imgs[i])
            # img = Image.open(path + imgs[i]).convert('L')
            img = img.resize((size, size), PIL.Image.ANTIALIAS)
            arr = np.array(img, dtype="float32")

            data[j, :, :, :] = arr / 255
            mean = np.mean(data)
            data -= mean
            label[j] = map_dict[type]
            j += 1
            label_num[type] += 1

        except Exception as e:
            continue
    return data,label

def split_train_test(path):
    dirs = os.listdir(path)
    kinds = len(dirs)
    train_set = "./data/train"
    test_set = "./data/test"

    if not os.path.exists(train_set):
        os.mkdir(train_set)
    if not os.path.exists(test_set):
        os.mkdir(test_set)

    for i in range(kinds):
        if not os.path.isdir(path + dirs[i]):
            continue
        flowers = os.listdir(path + dirs[i])

        flower_num = len(flowers)
        split_num = flower_num * 0.8

        for j in range(flower_num):
            img_file = flowers[j]
            location = path + dirs[i] + "/" + img_file
            if j < split_num:
                shutil.copyfile(location, train_set + "/" + img_file)
            else:
                shutil.copyfile(location, test_set + "/" + img_file)

# 分割训练集和测试集
# split_train_test("./data/src/")

