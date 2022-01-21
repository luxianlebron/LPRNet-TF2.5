import time
import numpy as np
from dataset import CHARS

def metric(test_dataset, model):
    """
        评价函数
    """
    tp, ep = 0, 0  # tp 为正确预测数量, ep 是错误预测数量
    start_time = time.time()
    for cur_batch, (test_imgs, test_labels) in enumerate(test_dataset):
        prebs = model(test_imgs)
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j]))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == (len(CHARS) - 1):
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(test_labels[i]):
                ep += 1
                continue
            if (np.asarray(test_labels[i]) == np.asarray(label)).all():
                tp += 1
            else:
                ep += 1     
    end_time = time.time()
    t = end_time - start_time  # 评估总时间
    acc = tp * 1.0 / (tp + ep)  # 准确率
    return acc, tp, ep, t