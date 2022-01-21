import os
import math
import cv2
import argparse
import random
import tensorflow as tf

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", 
            "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", 
            "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
             "新"]
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 
        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
        'W', 'X', 'Y', 'Z', 
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# 建立处理参数
def get_parser():
    parser = argparse.ArgumentParser(description='parameters to deal CCPD datasets')
    parser.add_argument('--ccpd_dir', default='', help='dir to CCPD dataset')
    parser.add_argument('--saved_dir', default='./ccpd_after_dealing', help='the saved folder of CCPD dataset after dealing')
    parser.add_argument('--split', default=0., type=float, help='split to train and test dataset')
    args = parser.parse_args()
    return args


def verify_img(img_file):
    if isinstance(img_file, str):
        if img_file.endswith(".jpg") \
                or img_file.endswith(".JPG") \
                or img_file.endswith(".PNG") \
                or img_file.endswith(".png"):
            return True
    return False
    
    
def fetch_plate(img):
    plate_str = ''
    inf = img.split('&')
    if len(inf) > 2:
        img_name = inf[-1]
        plate_ = img_name.split('-')[1]
        numbers = plate_.split('_')
        if len(numbers) < 7:
            return ''
        plate_str = provinces[int(numbers[0])]
        plate_str += ads[int(numbers[1])]
        plate_str += ads[int(numbers[2])]
        plate_str += ads[int(numbers[3])]
        plate_str += ads[int(numbers[4])]
        plate_str += ads[int(numbers[5])]
        plate_str += ads[int(numbers[6])]
    return plate_str


def fetch_box(img):
    box_ = []
    inf = img.split('&')
    if len(inf) > 2:
        img = img.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        left_up, right_bottom = [[int(eel) for eel in el.split('&')] for el in img[2].split('_')]
        box_.append((int(left_up[0] - 5), int(left_up[1])))
        box_.append((int(right_bottom[0]), int(right_bottom[1])))
    return box_


def fetch_plate_img(img_dir, saved_dir):
    """
    保存车牌图片
    :param img_dir:
    :return:
    """
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    count = 0
    for root, dirs, files in os.walk(img_dir):
        for img in files:
            if not verify_img(img):
                continue
            box = fetch_box(img)
            if len(box) == 0 or box[0][0] >= box[1][0] or box[0][1] >= box[1][1]:
                continue
            image = cv2.imread(os.path.join(root, img))
            if image is None:
                print(os.path.join(root, img))
                continue
            # 开始的y坐标:结束的y坐标,开始x:结束的x
            crop_img = image[int(box[0][1]):int(box[1][1]), int(box[0][0]): int(box[1][0])]
            try:
                crop_img = cv2.resize(crop_img, (94, 24))
            except:
                continue 
            count += 1
            plate = fetch_plate(img)
            cv2.imwrite(os.path.join(saved_dir, '{0}.jpg'.format(plate)), crop_img)
            print("Count:{0}, plate:{1}".format(count, plate))

def generate_train_eval(img_dir, train_dir, eval_dir, eval_percent=0.05):
    """
    生成训练-评估数据
    :param img_dir:
    :param train_dir:
    :param eval_dir:
    :param eval_percent: 评估数据占比%
    :return:
    """
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    train_num = 0
    eval_num = 0
    images = os.listdir(img_dir)
    img_num = len(images)
    val_num = math.ceil(img_num*eval_percent)
    for img in images:
        if not verify_img(img):
            continue
        print(os.path.join(img_dir, img))
        plate = img
        img = cv2.imread(os.path.join(img_dir, plate))
        if val_num > 0:
            cv2.imwrite(os.path.join(eval_dir, '{0}'.format(plate)), img)
            val_num -= 1
            eval_num += 1
            print(os.path.join(eval_dir, plate))
        else:
            train_num += 1
            cv2.imwrite(os.path.join(train_dir, '{0}'.format(plate)), img)            
    print("train: {0}张，eval: {1}张".format(train_num, eval_num))


if __name__ == '__main__':
    args = get_parser()
    fetch_plate_img(args.ccpd_dir, args.saved_dir)
    if args.split:
        generate_train_eval(args.saved_dir, './train_imgs', './test_imgs', args.split)