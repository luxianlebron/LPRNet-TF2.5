import tensorflow as tf
import numpy as np
import os

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
          'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z', 
          '-']

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}  


def tranform(img, img_size=None, aug=False, expand_dims=None):
    """
        对图片数据进行缩放和归一化操作；
        参数 aug [True, Flase]:
            控制是否对图片进行随机增加光照强度和饱和度；
        参数 expand_dims [None, integer]: 
            控制是否对图片数据增加维度。
    """
    if img_size is not None:
        img = tf.image.resize(img, img_size)
    img = img  / 255.
    if aug:
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_saturation(img, 0.5, 1.5)
    if expand_dims is not None:
        img = np.expand_dims(img, expand_dims)
    return img



def check(label):
    """
        检查label是否为异常值
        input: label
        return: True or False
    """
    if len(label) == 8:
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, please check it")
            return False
    else:
        return True



def build_labels_and_paths(data_dir):
    """
        生成保存图片数据的 paths list 和 labels
    """
    img_dirnames = []
    labels = []
    for cur_root, dirs, files in os.walk(data_dir): 
        for file in files:
            if not tf.io.read_file(tf.constant(os.path.join(cur_root, file))):
                continue
            label = []  # 用来保存单个图片中车牌字符的label
            license_plate, suffix = os.path.splitext(file)
            for c in license_plate:
                label.append(CHARS_DICT[c])
            if check(label) == False:
                assert 0, "Error label!"
            img_dirnames.append(os.path.join(cur_root, file))
            labels.append(label)
    return img_dirnames, labels


def load_imgs(path):
    raw = tf.io.read_file(path)
    image = tf.io.decode_jpeg(raw, channels=3)
    return image


def build_pipeline_from_path(data_dir, aug=False):
    """
        生成 tensorflow pipeline
    """
    img_dirnames, labels = build_labels_and_paths(data_dir)
    dataset = tf.data.Dataset.from_tensor_slices((img_dirnames, labels)).map(lambda x, y: \
                                    (load_imgs(x), y))
    return dataset
