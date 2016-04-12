"""Tests for aurora input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import tensorflow as tf
import aurora_dnn.input_data as input_data
import tensorflow_models_cifar10.compat as compat
import numpy as np


class AuroraInputTest(tf.test.TestCase):
  def _record(self, list):
    image_size = 128 * 128
    record = bytes(bytearray(list))
    expected = list
    return record, expected

  def img2list(self, url, labels, images):
    imgs_list = []
    imgs_expected = []
    for i, image in enumerate(images):
        path = os.path.join(url, str(labels[i]))
        path = path + '/' + image
        print(path)
        img = cv2.imread(path)
        _, g, _ = cv2.split(img)
        gr = cv2.resize(g, (256, 256))
        tmp = gr
        tmp = tmp[:, :, np.newaxis]
        tmp_record = np.ndarray.tolist(tmp)
        imgs_expected.append(np.ndarray.tolist(tmp))
        gr = gr.reshape((1, -1))
        img_list = np.ndarray.tolist(gr)
        record = [labels[i]] + img_list[0]
        records = bytes(bytearray(record))
        imgs_list.append(records)
    return imgs_list, imgs_expected


  def gen_bin_files(self, url):
    datas = ['03']
    for img_label in [4]:
        dir = url+str(img_label)
        for data in datas:
            files = [os.path.join(dir, f) for f in os.listdir(dir) if f[7:9] == data]
            files.sort()
            imgs_expected = []
            for i, image_path in enumerate(files):
                print(image_path)
                img = cv2.imread(image_path)
                _, g, _ = cv2.split(img)
                gr = cv2.resize(g, (128, 128))
                tmp = gr
                tmp = tmp[:, :, np.newaxis]
                imgs_expected.append(np.ndarray.tolist(tmp))
    return imgs_expected, len(imgs_expected)



  # def testSimple(self):
  #   url = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora2/'
  #   labels = [1, 2, 3, 4]
  #   images = ['N20031221G035621.bmp', 'N20031221G031111.bmp', 'N20031221G101531.bmp', 'N20031222G041721.bmp']
  #
  #   results, expectes = self.img2list(url, labels, images)
  #   i = len(results)
  #   contents = b"".join(results)
  #   expected = [expected for expected in expectes]
  #   filename = os.path.join('/tmp', "aurora_test.bin")
  #   open(filename, "wb").write(contents)
  #
  #   with self.test_session() as sess:
  #     q = tf.FIFOQueue(1024, [tf.string], shapes=())
  #     q.enqueue([filename]).run()
  #     q.close().run()
  #     result = input_data.read_aurora(q)
  #
  #     for i in range(4):
  #       key, label, uint8image = sess.run([
  #           result.key, result.label, result.uint8image])
  #       # print("the value of key"+key)
  #       # print(label)
  #       # print(len(uint8image), len(uint8image[0]), len(uint8image[0][1]))
  #       # print(uint8image)
  #       # print(expected[i])
  #       self.assertEqual("%s:%d" % (filename, i), compat.as_text(key))
  #       self.assertEqual(labels[i], label)
  #       self.assertAllEqual(expected[i], uint8image)
  #
  #     with self.assertRaises(tf.errors.OutOfRangeError):
  #       sess.run([result.key, result.uint8image])


  def testSimple2(self):
    # url = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora2/'
    # labels = [1, 2, 3, 4]
    # images = ['N20031221G035621.bmp', 'N20031221G031111.bmp', 'N20031221G101531.bmp', 'N20031222G041721.bmp']
    #
    # results, expectes = self.img2list(url, labels, images)
    # i = len(results)
    # contents = b"".join(results)
    # expected = [expected for expected in expectes]
    # filename = os.path.join('/tmp', "aurora_test.bin")
    # open(filename, "wb").write(contents)

    labels = 3
    expected, nums = self.gen_bin_files('/home/aurora/hdd/workspace/PycharmProjects/data/aurora2/')
    # filename = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora2/bin_files/type1_data21.bin'
    filename = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora2/bin_files_128/type3_data03.bin'
    with self.test_session() as sess:
      q = tf.FIFOQueue(10000, [tf.string], shapes=())
      q.enqueue([filename]).run()
      q.close().run()
      result = input_data.read_aurora(q)

      for i in range(nums):
        key, label, uint8image = sess.run([
            result.key, result.label, result.uint8image])
        # print("the value of key"+key)
        print(label)
        # print(len(uint8image), len(uint8image[0]), len(uint8image[0][1]))
        # print(uint8image)
        # print(expected[i])
        self.assertEqual("%s:%d" % (filename, i), compat.as_text(key))
        self.assertEqual(labels, label)
        self.assertAllEqual(expected[i], uint8image)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run([result.key, result.uint8image])


if __name__ == "__main__":
  tf.test.main()
