# encoding:UTF-8
import os
import tensorflow as tf
import input_data
import cv2
import numpy as np

tf.app.flags.DEFINE_string('directory', '/home/aurora/hdd/video/test_images/testrecords/',
                           'Directory to download data files and write the '
                           'converted result')
tf.app.flags.DEFINE_string('data_url', '/home/aurora/hdd/video/test_images/testnopeople256/',
                           'Directory to download data files and write the '
                           'converted result')
tf.app.flags.DEFINE_string('data_url2', '/home/aurora/hdd/video/test_images/testhavepeople256/',
                           'Directory to download data files and write the '
                           'converted result')

tf.app.flags.DEFINE_integer('validation_size', 0,
                            'Number of examples to separate from the training '
                            'data for the validation set.')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
  num_examples = labels.shape[0]
  # image is a four dim tensor
  if images.shape[0] != num_examples:
    raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]
  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())


def dense_to_one_hot(labels_dense, num_classes=2):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  # print('Extracting', filename)
  img = cv2.imread(filename[0])
  # print img.shape, img.dtype
  train_images = np.zeros([len(filename), img.shape[0], img.shape[1], img.shape[2]], dtype=np.uint8)
  for i, f in enumerate(filename):
      train_images[i, :, :, :] = cv2.imread(f)
  # print train_images.shape
  return train_images


def main(argv):
  # nopeople:0   people:1
  # Get the data.
  train_images_filename = [os.path.join(FLAGS.data_url, f) for f in os.listdir(FLAGS.data_url)]
  train_label_original = np.zeros([len(train_images_filename)], dtype=np.uint8)

  train_images_filename2 = [os.path.join(FLAGS.data_url2, f) for f in os.listdir(FLAGS.data_url2)]
  train_label_original2 = np.ones([len(train_images_filename2)], dtype=np.uint8)
  total_images = train_images_filename + train_images_filename2
  # Extract it into numpy arrays.
  train_images = extract_images(total_images)

  train_labels = np.zeros(len(train_images_filename + train_images_filename2))
  train_labels[len(train_images_filename):] = 1
  # print train_labels
  # print train_images.shape
  # train_labels = dense_to_one_hot(train_label_original)
  # convert_to(train_images, train_label_original, 'nopeople')
  # convert_to(train_images, train_label_original, 'havepeople')
  # train_labels = input_data.extract_labels(train_labels_filename)
  # Generate a validation set.
  # Convert to Examples and write the result to TFRecords.
  # convert_to(train_images, train_labels, 'train')
  convert_to(train_images, train_labels, 'test')
  # convert_to(validation_images, validation_labels, 'validation')
  # convert_to(test_images, test_labels, 'test')
if __name__ == '__main__':
  tf.app.run()

