import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import glob
import datetime
from tqdm import tqdm
import math
import sys

from model import efficientnet_b0 as B0_create_model
from model import efficientnet_b1 as B1_create_model
from model import efficientnet_b2 as B2_create_model
from model import efficientnet_b3 as B3_create_model
from model import efficientnet_b4 as B4_create_model
from model import efficientnet_b5 as B5_create_model
from model import efficientnet_b6 as B6_create_model
from model import efficientnet_b7 as B7_create_model

from utils import generate_ds

def test(num_classes, num_model, test_ds=None):
  # data_root = "/content/drive/MyDrive/HAU/efficientNet_tf/data/Swin_Transformer_kitchen"  # get data root path

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
  epochs = 1
  num_classes = num_classes
  freeze_layers = True
  initial_lr = 0.01

  log_dir = "/content/drive/MyDrive/HAU/efficientNet_tf/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  test_writer = tf.summary.create_file_writer(os.path.join(log_dir, "test"))

  # data generator with data augmentation
  # train_ds, val_ds, test_ds = generate_ds(data_root, im_height, im_width, batch_size)

  # create model
  model = model_dict[num_model](num_classes=num_classes)

  # load weights
  pre_weights_path = "/content/drive/MyDrive/HAU/efficientNet_tf/save_weights/efficientnet" + num_model +".h5"
  assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
  model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

  # freeze bottom layers
  if freeze_layers:
      for layer in model.layers:
            layer.trainable = False

  # model.summary()

  # custom learning rate curve
  def scheduler(now_epoch):
      end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
      rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
      new_lr = rate * initial_lr

      # writing lr into tensorboard

      return new_lr

  # using keras low level api for training
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
  optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

  test_loss = tf.keras.metrics.Mean(name=' test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name=' test_accuracy')

  @tf.function
  def test_step(test_images, test_labels):
      output = model(test_images, training=False)
      loss = loss_object(test_labels, output)

      test_loss.update_state(loss)
      test_accuracy.update_state(test_labels, output)

  for epoch in range(epochs):
      test_loss.reset_states()  # clear history info
      test_accuracy.reset_states()  # clear history info

      # test
      test_bar = tqdm(test_ds, file=sys.stdout)
      i =0
      for images, labels in test_bar:
        i += 1
        test_step(images, labels)

        print("test batch[{}] loss:{:.3f}, acc:{:.3f}".format(i,
                                                              test_loss.result(),
                                                              test_accuracy.result()))

      # writing test loss and acc
      with test_writer.as_default():
          tf.summary.scalar("loss", test_loss.result(), epoch)
          tf.summary.scalar("accuracy", test_accuracy.result(), epoch)


if __name__=="__main__":
  test()