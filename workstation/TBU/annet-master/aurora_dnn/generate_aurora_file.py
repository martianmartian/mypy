#encoding:UTF-8
import os
import numpy as np
from pysqlite2 import dbapi2 as sqlite
import cv2


class Indexer(object):
    def __init__(self, db):
        """初始化数据库的名称及词汇对象"""
        self.con = sqlite.connect(db)

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()

    def create_tables(self):
        """创建数据库表单"""
        self.con.execute('create table images(imid INTEGER PRIMARY KEY autoincrement, location, a_year, a_month, a_day, a_rgb, a_hour, a_min, a_sec, a_type, a_file BLOB, a_rfile BLOB)')
        self.db_commit()

    def insert_data(self, count, names, type, g, rg):
        sql = 'insert into images(imid, location, a_year, a_month, a_day, a_rgb, a_hour, a_min, a_sec, a_type, a_file, a_rfile) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
        self.con.execute(sql, (count, names[0], names[1], names[2], names[3], names[4], names[5], names[6], names[7], type, g, rg))
    # db = Indexer('/home/aurora/hdd/workspace/PycharmProjects/data/aurora2/all.db')
    # db.create_tables()


def parse_filename(file_name):
    # N20031221G041301
    filename_list = []
    filename_list.append(file_name[0])
    filename_list.append(file_name[1:5])
    filename_list.append(file_name[5:7])
    filename_list.append(file_name[7:9])
    filename_list.append(file_name[9])
    filename_list.append(file_name[10:12])
    filename_list.append(file_name[12:14])
    filename_list.append(file_name[14:])
    return filename_list


def img_data(file):
    file = cv2.imread(file)
    _, g, _ = cv2.split(file)
    # rg = cv2.resize(g, (256, 256))
    rg = cv2.resize(g, (128, 128))
    # cv2.imshow('test', file)
    # cv2.imshow('rgsize', rg)
    # cv2.waitKey(0)
    g = g.reshape((1, -1))
    rg = rg.reshape((1, -1))
    return g, rg


def get_count(url, key_word):
    files = [os.path.join(url, f) for f in os.listdir(url) if f[0:6] == key_word]
    # files = [os.path.join(url, f) for f in os.listdir(url) if f[0:6] == key_word]
    print files
    totalcount = 0
    for file in files:
        value = np.load(file)
        print value.shape
        totalcount += value.shape[0]
    print totalcount


def get_count2(url, keys):
    files = [os.path.join(url, f) for f in os.listdir(url) if f[10:12] == keys]
    # files = [os.path.join(url, f) for f in os.listdir(url) if f[0:6] == key_word]
    print files
    totalcount = 0
    for file in files:
        value = np.load(file)
        print file, value.shape
        totalcount += value.shape[0]
    print totalcount


def get_files(config_file):
    datas = []
    with open(config_file, 'r') as f:
        for lines in f:
            if lines.startswith('#'):
                continue
            temp = lines.split(',')
            temp = [t.strip() for t in temp]
            datas.append(temp)
        print datas
    return datas


def gen_file_list(url, keys):
    filelists = []
    for key in keys:
        for i in ['1', '2', '3', '4']:
            filename = 'type'+i+'_data'+key+'.npy'
            if i == '4' and key == '02':
                continue
            filelists.append(os.path.join(url, filename))

    filelists.sort()
    # print filelists
    return filelists


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
        imgs_expected.append(np.ndarray.tolist(tmp))

        gr = gr.reshape((1, -1))
        img_list = np.ndarray.tolist(gr)
        record = [labels[i]] + img_list[0]
        records = bytes(bytearray(record))
        imgs_list.append(records)
    return imgs_list, imgs_expected


def gen_bin_files(url, save_path):
    datas = ['21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '01', '02', '03', '12', '14', '16', '17', '18']
    # for img_label in [0, 1, 2, 3]:
    for img_label in [0, 1]:
        dir = url+str(img_label+2)
        for data in datas:
            filename = save_path+'/type'+str(img_label)+'_data'+data+'.bin'
            files = [os.path.join(dir, f) for f in os.listdir(dir) if f[7:9] == data]
            files.sort()
            imgs_expected = []
            imgs_list = []
            for i, image_path in enumerate(files):
                # print image_path
                img = cv2.imread(image_path)
                _, g, _ = cv2.split(img)
                # gr = cv2.resize(g, (256, 256))
                gr = cv2.resize(g, (256, 256))
                tmp = gr
                tmp = tmp[:, :, np.newaxis]
                imgs_expected.append(np.ndarray.tolist(tmp))
                gr = gr.reshape((1, -1))
                img_list = np.ndarray.tolist(gr)
                record = [img_label] + img_list[0]
                records = bytes(bytearray(record))
                imgs_list.append(records)
            contents = b"".join(imgs_list)
            open(filename, "wb").write(contents)


if __name__=='__main__':
    # url = '/home/aurora/hdd/workspace/data/aurora2/'
    # save_path = '/home/aurora/hdd/workspace/data/aurora2'
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora2/'
    save_path = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora2/bin_files_2_3_class_256*256'

    # IMAGE_SIZE = 256*256
    # IMAGE_SIZE = 128*128
    # datas = ['21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '01', '02', '03', '12', '14', '16', '17', '18']
    # for i in ['1', '2', '3', '4']:
    #     dir = url+i
    #     for data in datas:
    #         temp = save_path+'/type'+i+'_data'+data
    #         files = [os.path.join(dir, f) for f in os.listdir(dir) if f[7:9] == data]
    #         files.sort()
    #
    #         result = np.zeros((len(files), IMAGE_SIZE+1))
    #         index = 0
    #         for f in files:
    #             g, rg = img_data(f)
    #             result[index, 0] = int(i)
    #             result[index, 1:] = rg
    #             index += 1
    #         print result
    #         print result.shape
    #         np.save(temp, result)
    # get_count(url, 'type1_')  #18417
    # get_count(url, 'type1.')  #18417
    # keys = get_files('config.txt')
    # get_count2(url, keys[0])
    # gen_file_list(url, keys[0])
    gen_bin_files(url, save_path)
    # datas = ['21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '01', '02', '03', '12', '14', '16', '17', '18']
    # for data in datas:
    #     get_count2('/home/aurora/hdd/workspace/PycharmProjects/data/aurora2/npy_data/', data)
    #     print '----------------------------------------------'