#coding:utf-8

import os
import PIL
from PIL import Image
import numpy as np
import shutil


def load_data(path):
    imgs = os.listdir(path)
    num = len(imgs)
    map_dict = {"n04971313":0, "n11950345":1, "n11978233":2, "n12306717":3, "n12649065":4, "n12421683":5}

    size = 80
    data = np.empty((num, size, size, 1), dtype="float32")
    label = np.empty((num,), dtype="uint8")

    for i in range(num):
        try:
            img = Image.open(path + imgs[i]).convert('L')
            # arr = np.asarray(img, dtype="float32")
            img = img.resize((size, size), PIL.Image.ANTIALIAS)
            arr = np.array(img, dtype="float32")
            # arr = arr.reshape([size, size])
            data[i, :, :, 0] = arr
            label[i] = map_dict[(imgs[i].split('_')[0])]
        except Exception as e:

            c = label[i]
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
        # label = dirs[i]
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

