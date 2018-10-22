"""
make_img.py
read all the images and combine into a single one. 
"""
import os
import numpy as np
import cv2
import argparse
import random
from tqdm import tqdm


def bgr(img):
    return np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])


def bgr_chl_sum(img):
    return np.sum(img[:, :, 0]), np.sum(img[:, :, 1]), np.sum(img[:, :, 2])


def bgr_sum(img):
    return np.sum(img)


def av_hue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0])


def av_sat(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 1])


def hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])


def lum(img):
    return np.mean(np.sqrt(0.241 * img[:, :, 0] + .691 * img[:, :, 1] + .068 * img[:, :, 2]))


def lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return np.mean(lab[:, :, 0]), np.mean(lab[:, :, 1]), np.mean(lab[:, :, 2])


def rand(img):
    return random.random()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the downloaded head images", default="img", type=str)
    parser.add_argument("--out", help="The name of the output image", default="", type=str)
    parser.add_argument("--size", help="Size of each image in pixels", type=int, default=100)
    parser.add_argument("--ratio", help="Aspect ratio", nargs='+', type=int, default=[16, 9])
    parser.add_argument("--sort", help="Sort method",
                        choices=["none", "bgr", "bgr_sum", "av_hue", "av_sat", "lum", "lab", "rand",
                                 "pca_bgr", "pca_hsv", "pca_lab"],
                        type=str, default="bgr_sum")
    parser.add_argument("--rev_row", help="Whether to use the S-shaped alignment", action="store_true")
    parser.add_argument("--rev_sort", help="Sort in the reverse direction", action="store_true")

    args = parser.parse_args()

    # aspect ratio
    ratio = tuple(args.ratio)
    rw, rh = ratio

    pic_path = args.path
    files = list(os.walk(pic_path))[0][-1]
    try:
        files.remove("cache.pkl")
    except ValueError:
        pass

    num_friends = len(files)
    img_size = args.size, args.size
    rev = args.rev_row

    possible_wh = []
    print("Calculating grid size...")
    for width in range(1, num_friends):
        height = num_friends // width
        possible_wh.append((width, height))

    best_wh = min(possible_wh, key=lambda x: ((x[0] / x[1]) - (rw / rh)) ** 2)

    print("Calculated grid size based on your aspect ratio:", best_wh)
    print("Note", num_friends - best_wh[0] * best_wh[1], "of your friends will be thrown away from the collage")

    result_grid = best_wh

    imgs = []
    for i, img_file in tqdm(enumerate(files), total=len(files), desc="[Reading images]", unit="imgs"):
        try:
            img = cv2.resize(cv2.imread(os.path.join(pic_path, img_file)), img_size, interpolation=cv2.INTER_CUBIC)
        except:
            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        imgs.append(img)

    print("Sorting images...")
    if args.sort.startswith("pca_"):
        sort_function = eval(args.sort[args.sort.find('_') + 1:])
        from sklearn.decomposition import PCA

        pca = PCA(1)
        img_keys = pca.fit_transform(np.array(list(map(sort_function, imgs))))[:, 0]
    elif args.sort == "none":
        img_keys = np.array(list(range(0, num_friends)))
    else:
        sort_function = eval(args.sort)
        img_keys = np.array(list(map(sort_function, imgs)))

    sorted_imgs = np.array(imgs)[np.argsort(img_keys)]
    if args.rev_sort:
        sorted_imgs = list(reversed(sorted_imgs))

    for i in tqdm(range(result_grid[1]), desc="[Merging]", unit="imgs"):
        if rev and i % 2 == 1:
            img_f = sorted_imgs[(i + 1) * result_grid[0] - 1]
            for j in range(result_grid[0] - 2, -1, -1):
                img_t = sorted_imgs[i * result_grid[0] + j]
                img_f = np.append(img_f, img_t, axis=1)
            if i == 0:
                combined_img = img_f
            else:
                combined_img = np.append(combined_img, img_f, axis=0)
        else:
            img_f = sorted_imgs[i * result_grid[0]]
            for j in range(1, result_grid[0]):
                img_t = sorted_imgs[i * result_grid[0] + j]
                img_f = np.append(img_f, img_t, axis=1)
            if i == 0:
                combined_img = img_f
            else:
                combined_img = np.append(combined_img, img_f, axis=0)
    if len(args.out) == 0:
        cv2.imwrite('result-{}.png'.format(args.sort), combined_img)
    else:
        cv2.imwrite(args.out, combined_img)
