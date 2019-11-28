import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


def result():
    with open('result.piclke', 'rb') as f:
        result_data = pkl.load(f)
    return result_data


def save():
    data = result()
    res_loss = []
    res_acc = []

    with open('result.txt', 'w') as f:
        for line in data:
            line = f'{line}'
            f.write(f'{line}\n')

            loss = float(line[line.find('s:')+2:line.find(' |')])
            acc = float(line[line.find('y:')+2:line.find('\'}')])

            res_loss.append(loss)
            res_acc.append(acc)

    with open('res_loss.pickle', 'wb') as f:
        pkl.dump(res_loss, f)
    with open('res_acc.pickle', 'wb') as f:
        pkl.dump(res_acc, f)


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


def main():
    plot_loss()
    plot_acc()


if __name__ == '__main__':
    main()
