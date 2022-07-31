import os
import sys
import math
import datetime
import sys
sys.path.append('/content/drive/MyDrive/HAU/efficientNet_tf')

import tensorflow as tf
from tqdm import tqdm

from model import efficientnet_b0 as B0_create_model
from model import efficientnet_b1 as B1_create_model
from model import efficientnet_b2 as B2_create_model
from model import efficientnet_b3 as B3_create_model
from model import efficientnet_b4 as B4_create_model
from model import efficientnet_b5 as B5_create_model
from model import efficientnet_b6 as B6_create_model
from model import efficientnet_b7 as B7_create_model

from utils import generate_ds_TVT

assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"


def train(num_classes, num_model):
    data_root = "/content/drive/MyDrive/HAU/efficientNet_tf/data/dataset"  # get data root path

    if not os.path.exists("/content/drive/MyDrive/HAU/efficientNet_tf/save_weights"):
        os.makedirs("/content/drive/MyDrive/HAU/efficientNet_tf/save_weights")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}

    model_dict = {"B0": B0_create_model,
              "B1": B1_create_model,
              "B2": B2_create_model,
              "B3": B3_create_model,
              "B4": B4_create_model,
              "B5": B5_create_model,
              "B6": B6_create_model,
              "B7": B7_create_model}

    num_model = num_model
    im_height = im_width = img_size[num_model]
    batch_size = 16
    epochs = 30
    num_classes = num_classes
    freeze_layers = True
    initial_lr = 0.01

    log_dir = "/content/drive/MyDrive/HAU/efficientNet_tf/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # data generator with data augmentation
    train_ds, val_ds, test_ds = generate_ds_TVT(data_root, im_height, im_width, batch_size)

    # create model
    model = model_dict[num_model](num_classes=num_classes)

    # load weights
    pre_weights_path = '/content/drive/MyDrive/HAU/efficientNet_tf/efficientnetb' + num_model[-1] + '.h5'
    print(pre_weights_path)
    assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    # freeze bottom layers
    if freeze_layers:
        unfreeze_layers = ["top_conv", "top_bn", "predictions"]
        for layer in model.layers:
            if layer.name not in unfreeze_layers:
                layer.trainable = False
            else:
                print("training {}".format(layer.name))

    # model.summary()

    # custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * initial_lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # using keras low level api for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            loss = loss_object(train_labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(train_labels, output)

    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # update learning rate
        optimizer.learning_rate = scheduler(epoch)

        # validate
        val_bar = tqdm(val_ds, file=sys.stdout)
        for images, labels in val_bar:
            val_step(images, labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "/content/drive/MyDrive/HAU/efficientNet_tf/save_weights/efficientnet" + num_model +".h5"
            model.save_weights(save_name, save_format="h5")
    return test_ds

if __name__ == '__main__':
    train()
