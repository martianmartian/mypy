import numpy as np
import struct
import cv2
import os
import tensorflow as tf


def img2list(url, labels, images):
    imgs_list = []
    imgs_expected = []
    for i, image in enumerate(images):
        path = os.path.join(url, str(labels[i]))
        path = path + '/' + image
        print path
        img = cv2.imread(path)
        _, g, _ = cv2.split(img)
        gr = cv2.resize(g, (256, 256))
        tmp = gr
        tmp = tmp[:, :, np.newaxis]
        tmp_record = np.ndarray.tolist(tmp)
        print tmp_record
        imgs_expected.append(np.ndarray.tolist(tmp))

        gr = gr.reshape((1, -1))
        img_list = np.ndarray.tolist(gr)
        record = [labels[i]] + img_list[0]
        records = bytes(bytearray(record))
        imgs_list.append(records)
    return imgs_list, imgs_expected


if __name__=='__main__':
    # # grades = np.load('/tmp/cifar10_data/cifar-10-batches-bin/data_batch_1.bin')
    # with open('/tmp/cifar10_data/cifar-10-batches-bin/data_batch_1.bin', 'rb') as f:
    #     # newFileByteArray = bytearray(newFileBytes)
    #     # newFile.write(newFileByteArray)
    #     # # data = f.read()
    #     data = f.read(1)
    #     # data = f.read(32*32*3+1)
    #     text = data.decode('utf-8')
    #     print text
    #     data2 = f.read(32*32*3+1)
    #     text2 = data2.decode('utf-8')
    #     print text2
    #     # struct.unpack("iiiii", fileContent[:20])
    # #  dt = np.dtype([('time', [('min', int), ('sec', int)]),('temp', float)])
    # # np.fromfile('/tmp/cifar10_data/cifar-10-batches-bin/data_batch_1.bin')

    url = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora2/'
    labels = [1, 2, 3, 4]
    images = ['N20031221G035621.bmp', 'N20031221G031111.bmp', 'N20031221G101531.bmp', 'N20031222G041721.bmp']

    results, expectes = img2list(url, labels, images)
    contents = b"".join([record for record in results])
    expected = [expected for expected in expectes]
    filename = os.path.join('/tmp', "aurora_test.bin")
    open(filename, "wb").write(contents)