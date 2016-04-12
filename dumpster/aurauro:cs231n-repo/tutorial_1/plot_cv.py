import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = np.load('accuracy.npy')
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    print data
    print mean
    print std

    fig, ax = plt.subplots(nrows=1, sharex=True)
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    # x = np.array([1, 3, 5, 10, 20, 50, 100])
    ax.set_xlim((-1, 7))
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(['1', '3', '5', '10', '20', '50', '100'])
    y = mean
    for i in range(5):
        plt.scatter(x, data[i], color='r')
    plt.errorbar(x, y, yerr=std, fmt='-o')
    ax.set_title('Cross Validate')
    plt.show()