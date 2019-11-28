import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


def load_result():
    with open('result.pickle', 'rb') as f:
        result_data = pkl.load(f)
    return result_data


def save_pkl():
    data = load_result()
    res_loss = []
    res_acc = []
    res_each_num_per = []
    save_name = ['res_loss', 'res_acc', 'res_each_num_per']

    for line in data:
        res_loss.append(line[0][0])
        res_acc.append(line[0][1])
        res_each_num_per.append(line[1])

    for idx in range(3):
        file_name = f'{save_name[idx]}'
        with open(f'{file_name}.pickle', 'wb') as f:
            if idx == 0:
                pkl.dump(res_loss, f)
            elif idx == 1:
                pkl.dump(res_acc, f)
            else:
                pkl.dump(res_each_num_per, f)


def save_txt():
    data = load_result()
    with open('result.txt', 'w') as f:
        ratio = 0
        for line in data:
            loss = line[0][0]
            acc = line[0][1]
            each_num_per = line[1]

            line = f'Noise Ratio:{ratio} | Loss:{loss}, Accuracy:{acc},' \
                   f' Correct answer rate for each number:{each_num_per}\n'
            f.write(line)
            ratio += 1


def open_file():
    with open('res_loss.pickle', 'rb') as f:
        loss = pkl.load(f)
    with open('res_acc.pickle', 'rb') as f:
        acc = pkl.load(f)
    with open('res_each_num_per.pickle', 'rb') as f:
        each_num_per = pkl.load(f)
    return loss, acc, each_num_per


def plot_loss():
    with open('res_loss.pickle', 'rb') as f:
        loss = pkl.load(f)
    plt.plot(loss, label='Test loss')
    plt.xlabel('Noise Ratio')
    plt.ylabel('loss')
    plt.title('Result')
    plt.show()


def plot_acc():
    with open('res_acc.pickle', 'rb') as f:
        acc = pkl.load(f)
    plt.plot(acc, label='Test accuracy')
    plt.xlabel('Noise Ratio')
    plt.ylabel('accuracy')
    plt.title('Result')
    plt.show()


def plot_num():
    count = 0
    nrows = 3
    ncols = 4
    plt.figure(figsize=(12, 8))
    with open('res_each_num_per.pickle', 'rb') as f:
        each_num_per = pkl.load(f)

    for num in range(10):
        correct_rate = []
        for ratio in each_num_per:
            correct_rate.append(ratio[num])

        count += 1
        plt.subplot(nrows, ncols, count)
        plt.plot(correct_rate)
        tmp = 'Number:' + str(num)
        plt.title(tmp)

    plt.tight_layout()
    plt.show()


def plot_num_loop():
    with open('res_each_num_per.pickle', 'rb') as f:
        each_num_per = pkl.load(f)

    for num in range(10):
        correct_rate = []
        for ratio in each_num_per:
            correct_rate.append(ratio[num])

        plt.plot(correct_rate)
        tmp = 'Number:' + str(num)
        plt.title(tmp)
        plt.show()


def main():
    plot_loss()
    plot_acc()
    plot_num()
    plot_num_loop()


if __name__ == '__main__':
    main()
