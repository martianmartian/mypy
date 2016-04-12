# encoding:UTF-8
# convert video to image
import cv2
import numpy as np
import os

def create_floders(urls):
    for dir in urls:
        os.mkdir(dir+'_npy')

if __name__=='__main__':
    url = '/home/aurora/hdd/video/'
    # filenames = '/home/aurora/hdd/video/20150916/ch01_00000000066000000.mp4'
    urls = [os.path.join(url, dir) for dir in os.listdir(url) if not 'npy' in dir]
    for video_url in urls:
        save_urls = video_url+'_npy/'
        print save_urls
        videos = [os.path.join(video_url, video) for video in os.listdir(video_url)]
        i = 0
        for video in videos:
            # filename = video.split('/')[-1][0:-4]
            # save_dir = save_urls + filename
            print video
            cap = cv2.VideoCapture(video)
            while True:
                # frame = cv.QueryFrame(cap)
                ret, im = cap.read()
                # blur = cv2.GaussianBlur(im, (0, 0), 5)
                # cv2.imshow('video test', im)
                # frames.append(im)
                if not ret:
                    break
                cv2.imwrite(save_urls+'pic{:>05}.jpg'.format(i), im)
                key = cv2.waitKey(10)
                i += 1
            cap.release()
        cv2.destroyAllWindows()