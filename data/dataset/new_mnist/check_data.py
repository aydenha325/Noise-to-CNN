import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


def img_show(idx, noise_ratio=0):
    if noise_ratio:
        file_name = f'_(noised, {noise_ratio})'
    else:
        file_name = ''

    with open(f'mnist_test_image{file_name}.piclke', 'rb') as f:
        test_images = pkl.load(f)
    with open('mnist_test_label.piclke', 'rb') as f:
        test_labels = pkl.load(f)

    img = np.array(test_images)
    img = img.reshape(10000, 28, 28)

    print('answer :', test_labels[idx])
    print('noise ratio :', noise_ratio)
    plt.imshow(img[idx], cmap='Greys', interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    idx = 4870
    noise_ratio = 15
    img_show(idx, noise_ratio)
