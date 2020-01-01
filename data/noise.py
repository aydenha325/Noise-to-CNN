import os
import copy
import random as rd
import pickle as pkl


# 데이터셋 이미지와 레이블을 각각 리스트로 반환
def read_dataset():
    test_labels = []
    test_images = []

    # 파일 읽기
    current_dir = os.getcwd()
    with open(f'{current_dir}\dataset\mnist_test.txt', 'r') as f:
        data = f.read()
    data_list = data.splitlines()

    # 읽어들인 파일 리스트로 변환
    tmp_test_images = []
    for line in data_list:
        test_labels.append(line[0])
        tmp_test_images.append(line[2:])

    # 리스트 형변환
    test_labels = list(map(int, test_labels))
    for line in tmp_test_images:
        new_line = line.split(',')
        test_images.append(list(map(int, new_line)))

    return test_images, test_labels


# 데이터셋 초깃값 구성
def init_dataset():
    test_images, test_labels = read_dataset()
    current_dir = os.getcwd()
    with open(f'{current_dir}\\dataset\\new_mnist\\mnist_test_label.piclke', 'wb') as f:
        pkl.dump(test_labels, f)
    with open(f'{current_dir}\\dataset\\new_mnist\\mnist_test_image.piclke', 'wb') as f:
        pkl.dump(test_images, f)
    return test_images


# 초기 데이터셋 불러오기
def load_dataset():
    current_dir = os.getcwd()
    with open(f'{current_dir}\\dataset\\new_mnist\\mnist_test_image.piclke', 'rb') as f:
        test_images = pkl.load(f)
    return test_images


def make_noise(test_images, noise_ratio):
    noised_image = copy.deepcopy(test_images)

    idx_0d = 0
    for number in test_images:
        idx_1d = 0
        for pixel in number:
            ratio = rd.randint(0, 99)
            if noise_ratio > ratio:
                noise_pixel = rd.randint(0, 255)
                noised_image[idx_0d][idx_1d] = noise_pixel
            idx_1d += 1
        idx_0d += 1

    return noised_image


# 노이즈 이미지 데이터 저장
def save_noised_dataset(noised_images, noise_ratio):
    current_dir = os.getcwd()
    with open(f'{current_dir}\\dataset\\new_mnist\\mnist_test_image_(noised, {noise_ratio}).piclke', 'wb') as f:
        pkl.dump(noised_images, f)


def main_noise(noise_ratio):
    # noise_ratio : 이미지에 부여할 노이즈 비율
    print(f'noise ratio : {noise_ratio}')

    try:
        test_images = load_dataset()
    except:
        test_images = init_dataset()
        print('dataset saved successfully.')

    print('generating noise...')
    noised_images = make_noise(test_images, noise_ratio)
    print('success')

    print('saving dadaset...')
    save_noised_dataset(noised_images, noise_ratio)
    print('noised dataset saved successfully.\n')


if __name__ == '__main__':
    for ratio in range(100):
	    main_noise(ratio)
