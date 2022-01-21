from dataset import CHARS
import numpy as np
import cv2


def predict(img_dir, model):
    """
        预测函数
    """
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (94, 24))
    image = image / 255.
    image = np.expand_dims(image, 0)

    logit = model(image)
    pre_label = list()
    for j in range(logit.shape[2]):
        pre_label.append(np.argmax(logit[:, :, j]))
    no_repeat_blank_label = list()
    pre_c = pre_label[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in pre_label:
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == (len(CHARS) - 1):
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    text = ''.join([CHARS[i] for i in no_repeat_blank_label])
    # # 将预测字符添加到图片上并打印图片
    # AddText = image.copy()
    # cv2.putText(AddText, text, (2,10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # text_image = np.hstack([image, AddText])
    # cv2.imshow('prediction:', text_image)
    # cv2.waitKey()
    # cv2.destoryAllWindows()
    return text