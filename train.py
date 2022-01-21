import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataset import (build_pipeline_from_path, 
                        tranform, 
                        CHARS
                    )
from model import metric, LPRNet
import tensorflow as tf
import numpy as np
import argparse
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train model')
    parser.add_argument('--epochs', default=300, help='epochs to train the model')
    parser.add_argument('--img_size', default=(24, 94), help='the size of images for training')
    parser.add_argument('--train_img_dir', default="", help='the train images path')
    parser.add_argument('--test_img_dir', default="", help='the test images path, default None')
    parser.add_argument('--split', default=0, type=float, help='if test_img_dir is None, the parameter must be given')
    parser.add_argument('--initial_lr', default=0.001, help='the initial learning rate')
    parser.add_argument('--dropout_rate', default=0.5, help='the dropout rate for layer')
    parser.add_argument('--lpr_len', default=7, help='the max length of license plate number')
    parser.add_argument('--train_batch_size', default=128, help='training batch size')
    parser.add_argument('--test_batch_size', default=128, help='testing batch size')
    parser.add_argument('--saved_model_folder', default="./saved_model", help='Location to save model')
    parser.add_argument('--pretrained_model', default="", help='pretrained base model')
    args = parser.parse_args()
    return args
                                

# 训练模型
def train():
    args = get_parser()

    if not os.path.exists(args.saved_model_folder):
        os.mkdir(args.saved_model_folder)

    # 实例化模型
    model = LPRNet(num_chars=len(CHARS), dropout_rate=args.dropout_rate)
    print("\n ********** Successful to build network! ********** \n")

    # 加载预训练模型
    if args.pretrained_model:
        model.load_weights(args.pretrained_model)
        print("\n ********** Successful to load pretrained model! ********** \n")

    # 优化器使用 RMSprop or Adam, 优先使用 Adam
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.initial_lr, momentum=args.momentum)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.initial_lr)

    # 加载数据集
    train_dataset = build_pipeline_from_path(args.train_img_dir)
    num_train_imgs = len(list(train_dataset.as_numpy_iterator()))

    if not args.test_img_dir:
        num_test_imgs = int(args.split * num_train_imgs)
        num_train_imgs -= num_test_imgs
        test_dataset = train_dataset.take(num_test_imgs)
        train_dataset = train_dataset.skip(num_test_imgs)
    else:
        test_dataset = build_pipeline_from_path(args.test_img_dir)
    test_dataset = test_dataset.map(lambda x, y: \
                                    (tranform(x, img_size=args.img_size), tf.cast(y, tf.int32)), 
                                    num_parallel_calls=4)
    test_dataset = test_dataset.batch(args.test_batch_size)
    

    train_dataset = train_dataset.map(lambda x, y: \
                                    (tranform(x, img_size=args.img_size), tf.cast(y, tf.int32)), 
                                    num_parallel_calls=4)
    train_dataset = train_dataset.shuffle(buffer_size=256)
    train_dataset = train_dataset.batch(args.train_batch_size)
    train_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()

    # 模型训练
    top_acc = 0.
    for cur_epoch in range(1, args.epochs + 1):
        batch = 0
        for batch_index, (train_imgs, train_labels) in enumerate(train_dataset):
            start_time = time.time()
            with tf.GradientTape() as tape:
                train_logits = model(train_imgs) #[N,66,18]
                train_logits = tf.transpose(train_logits, [2, 0, 1]) #[18,N,66]
                logits_shape = train_logits.shape
                logit_length = tf.fill([logits_shape[1]], logits_shape[0]) #(N)
                label_length = tf.fill([logits_shape[1]], args.lpr_len)
                loss = tf.nn.ctc_loss(labels=train_labels,
                                      logits=train_logits,
                                      label_length=label_length,
                                      logit_length=logit_length,
                                      logits_time_major=True,
                                      blank_index=len(CHARS) - 1)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
            end_time = time.time()
            batch = batch + int(tf.shape(train_imgs)[0])
            print('\r' + "Epoch {0}/{1} || ".format(cur_epoch, args.epochs) 
                  + "Batch {0}/{1} || ".format(batch, num_train_imgs)
                  + "Loss:{} || ".format(loss)
                  + "A Batch time:{0:.4f}s || ".format(end_time - start_time)
                  + "Learning rate:{0:.8f} || ".format(optimizer.lr.numpy().item()), end=''*20)
        acc, tp, tp_error, t = metric(test_dataset, model)
        print("\n******* Prediction accuracy: {0}/{1} || Acc:{2:.2f}%".format(tp, tp + tp_error, acc*100))
        print("******* Test speed: {}s 1/{}\n".format(t / (tp + tp_error), tp + tp_error))
        # 保存模型
        if acc >= top_acc:
            top_acc = acc
            model.save(args.saved_model_folder, save_format='tf')

    # 将.pb模型转为.tflite
    top_acc = 0.982
    cvtmodel = tf.keras.models.load_model(args.saved_model_folder)
    converter = tf.lite.TFLiteConverter.from_keras_model(cvtmodel)
    tflite_model = converter.convert()
    with open('model' + '{}'.format(np.around(top_acc * 100)) + '.tflite', "wb") as f:
        f.write(tflite_model)
    print("\n ********** Successful to convert tflite model! ********** \n")

if __name__ == '__main__':
    train()