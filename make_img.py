'''
make_img.py
read all the images and combine into a single one. 
'''
import os
import numpy as np
import cv2
import argparse
import random
from tqdm import tqdm


def bgr_sum(img):
    return np.sum(img)


def av_hue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0])


def hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])
    return h, s, v


def lum(img):
    return np.mean(np.sqrt(0.241 * img[:, :, 0] + .691 * img[:, :, 1] + .068 * img[:, :, 2]))


def rand(img):
    return random.random()


if __name__ == "__main__":
    # 获取下载的所有图片
    pic_path = 'img'
    files = list(os.walk(pic_path))[0][-1]
    try:
        files.remove("cache.pkl")
    except:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="Size of each image", type=int, default=100)
    parser.add_argument("--ratio", help="Aspect ratio", nargs='+', type=int, default=[16, 9])
    parser.add_argument("--sort", help="Sort method.", choices=["bgr_sum", "av_hue", "hsv", "lum", "rand"],
                        type=str, default="bgr_sum")

    args = parser.parse_args()

    # aspect ratio
    ratio = tuple(args.ratio)
    sort_key = eval(args.sort)
    rw, rh = ratio

    num_friends = len(files)
    img_size = args.size, args.size

    possible_wh = []
    print("Calculating grid size...")
    for width in range(rw, num_friends):
        height = num_friends // width
        possible_wh.append((width, height))

    best_wh = min(possible_wh, key=lambda x: ((x[0] / x[1]) - (rw / rh)) ** 2)

    print("Calculated grid size based on your aspect ratio:", best_wh)
    print("Note", num_friends - best_wh[0] * best_wh[1], "of your friends will be thrown away from the collage")

    result_grid = best_wh

    sorted_imgs = []
    for i, img_file in tqdm(enumerate(files), desc="[Reading images]", unit="imgs"):
        try:
            img = cv2.resize(cv2.imread(os.path.join(pic_path, img_file)), img_size, interpolation=cv2.INTER_CUBIC)
        except:
            # 如果读不出来图像，就用黑色块代替
            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

        sorted_imgs.append((img, sort_key(img)))

    sorted_imgs.sort(key=lambda x: x[1])
    sorted_imgs = [a for a, b in sorted_imgs]

    for i in tqdm(range(result_grid[1]), desc="[Processing]", unit="imgs"):
        img_f = sorted_imgs[i * result_grid[0]]
        for j in range(1, result_grid[0]):
            img_t = sorted_imgs[i * result_grid[0] + j]
            img_f = np.append(img_f, img_t, axis=1)
        if i == 0:
            combined_img = img_f
        else:
            combined_img = np.append(combined_img, img_f, axis=0)

    cv2.imwrite('result.png', combined_img)
