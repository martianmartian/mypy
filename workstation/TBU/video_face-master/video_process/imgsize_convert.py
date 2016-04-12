import cv2
import os

if __name__=='__main__':
    '''
        resize image
    '''
    url = '/home/aurora/hdd/video/test_images/testhavepeople/'
    files = [os.path.join(url, f) for f in os.listdir(url)]
    # save_url = '/home/aurora/hdd/video/20150918_npy_128/'
    save_url = '/home/aurora/hdd/video/test_images/testhavepeople256/'
    for i, f in enumerate(files):
      img = cv2.imread(f)
      img2 = cv2.resize(img, (256, 256))
      cv2.imwrite(save_url+'pic{:>05}.jpg'.format(i), img2)
      # cv2.imshow('resized img', img2)
      # cv2.waitKey(0)

    url2 = '/home/aurora/hdd/video/test_images/testnopeople/'
    files2 = [os.path.join(url2, f) for f in os.listdir(url2)]
    # save_url = '/home/aurora/hdd/video/20150918_npy_128/'
    save_url2 = '/home/aurora/hdd/video/test_images/testnopeople256/'
    for i, f in enumerate(files2):
      img = cv2.imread(f)
      img2 = cv2.resize(img, (256, 256))
      cv2.imwrite(save_url2+'pic{:>05}.jpg'.format(i), img2)