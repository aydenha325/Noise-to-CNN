'''
CNN 클래스 신경망 구조 구현 부분 코드(CNN, DATA)와 학습코드 등 일부분은
'코딩셰프의 3분 딥러닝 케라스맛'책의 예제 코드를 참고했습니다
https://github.com/jskDr/keraspp
'''

import os
import random
import pickle as pkl
import keras
from keras import models, layers
from keras import backend
from keras import datasets
from keras import utils
import numpy as np
import matplotlib.pyplot as plt
from func import *


class CNN(models.Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(layers.Conv2D(32, kernel_size=(3, 3),
                               activation='relu',
                               input_shape=input_shape))
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(layers.Dropout(0.25))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation='relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer='rmsprop',
                     metrics=['accuracy'])


# 데이터셋 구성
class DATA():
    def __init__(self):
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        img_rows, img_cols = x_train.shape[1:]

        if backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test


# 분류 CNN 학습 및 테스트
def train_and_test():
    batch_size = 128
    epochs = 10

    data = DATA()
    model = CNN(data.input_shape, data.num_classes)

    history = model.fit(data.x_train, data.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2)

    score = model.evaluate(data.x_test, data.y_test)
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights('mnist_cnn_weights.h5')

    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()


# 노이즈 데이터셋 로드
def noise_data(ratio):
    if ratio:
        file_name = f'_(noised, {ratio})'
    else:
        file_name = ''

    current_dir = os.getcwd()
    with open(f'{current_dir}\\data\\dataset\\new_mnist\\mnist_test_image{file_name}.piclke', 'rb') as f:
        x_test = pkl.load(f)
        x_test = np.array(x_test)
    with open(f'{current_dir}\\data\\dataset\\new_mnist\\mnist_test_label.piclke', 'rb') as f:
        y_test = pkl.load(f)
        y_test = np.array(y_test)

    return x_test, y_test


# 노이즈 인식 테스트
def test_noise(ratio, sample):
    data = DATA()
    model = CNN(data.input_shape, data.num_classes)
    model.load_weights('mnist_cnn_weights.h5')

    x_test, y_test = noise_data(ratio)

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32') / 255.
    y_test = utils.np_utils.to_categorical(y_test)

    score = model.evaluate(x_test, y_test, verbose=0)
    print()
    print('Noise ratio:', ratio)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    res_score = score[0], score[1]

    # 정답, 오답 확인을 위한 레이블 생성
    predicted_result = model.predict(x_test)
    predicted_labels = np.argmax(predicted_result, axis=1)
    test_labels = np.argmax(y_test, axis=1)

    '''
    # 데이터 인덱스 정답 체크
    idx = 1000
    print('\nLabel :', y_test[idx])
    print('Predict :', model.predict_classes(x_test[idx].reshape((1, 28, 28, 1))))
    plt.imshow(x_test[idx].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
    '''

    # 정답, 오답 확인 및 숫자별 인식률 확인
    correct_result = []
    wrong_result = []
    each_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    each_num_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for iscorrect in range(2):
        if not iscorrect:
            result_list = correct_result
        else:
            result_list = wrong_result

        for n in range(0, len(test_labels)):
            if not iscorrect:
                if predicted_labels[n] == test_labels[n]:
                    result_list.append(n)
                    each_num[test_labels[n]] = each_num[test_labels[n]]+1
                    each_num_correct[test_labels[n]] = each_num_correct[test_labels[n]]+1
            else:
                if predicted_labels[n] != test_labels[n]:
                    result_list.append(n)
                    each_num[test_labels[n]] = each_num[test_labels[n]]+1

        if sample:
            samples = random.choices(population=result_list, k=16)
            count = 0
            nrows = ncols = 4
            plt.figure(figsize=(12, 8))

            for n in samples:
                count += 1
                plt.subplot(nrows, ncols, count)
                plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
                tmp = "Label:" + str(test_labels[n]) + ", Prediction:" + str(predicted_labels[n])
                plt.title(tmp)

            plt.tight_layout()
            plt.show()

    res_each_num = []
    for num in range(10):
        percent_num = (each_num_correct[num] / each_num[num]) * 100
        res_each_num.append(percent_num)

    return res_score, res_each_num


def main(ratio=0, sample=False, loop=False):
    result = []
    if loop:
        for ratio in range(100):
            result.append(test_noise(ratio, sample))
    else:
        result = test_noise(ratio, sample)
    return result


# 시험 결과 저장
def _save():
    sample = False
    loop = True

    res = main(0, sample, loop)
    with open('result.pickle', 'wb') as f:
        pkl.dump(res, f)
    print('\nDone!')


if __name__ == '__main__':
    ratio = 0
    sample = True
    res = main(ratio, sample)
    print(res)
