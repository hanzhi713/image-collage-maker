import os
from os.path import *
import numpy as np
import cv2
import argparse
import random
from tqdm import tqdm


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


def av_lum(img):
    return np.mean(np.sqrt(0.241 * img[:, :, 0] + 0.691 * img[:, :, 1] + 0.068 * img[:, :, 2]))


def pca_lum(img):
    return np.sqrt(0.241 * img[:, :, 0] + 0.691 * img[:, :, 1] + 0.068 * img[:, :, 2]).flatten()


def pca_sat(img):
    return (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]).flatten()


def pca_bgr(img):
    return img.flatten()


def pca_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV).flatten()


def pca_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).flatten()


def pca_hue(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0].flatten()


def pca_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()


def rand(img):
    return random.random()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the downloaded head images",
                        default=join(dirname(__file__), "img"), type=str)
    parser.add_argument("--out", help="The name of the output image", default="", type=str)
    parser.add_argument("--size", help="Size of each image in pixels", type=int, default=100)
    parser.add_argument("--ratio", help="Aspect ratio", nargs='+', type=int, default=[16, 9])
    parser.add_argument("--sort", help="Sort methods",
                        choices=["none", "bgr_sum", "av_hue", "av_sat", "av_lum", "rand",
                                 "pca_bgr", "pca_hsv", "pca_lab", "pca_gray", "pca_lum", "pca_sat", "pca_hue",
                                 "tsne_bgr", "tsne_hsv", "tsne_lab", "tsne_gray", "tsne_lum", "tsne_sat", "tsne_hue",
                                 "umap_bgr", "umap_hsv", "umap_lab", "umap_gray", "umap_lum", "umap_sat", "umap_hue",
                                 ],
                        type=str, default="bgr_sum")
    parser.add_argument("--collage", type=str, default="",
                        help="If you want to fit an image, specify the image path here")
    parser.add_argument("--copt", type=str, default="lab", choices=["hsv", "bgr", "lab"],
                        help="Methods to use when fitting an image")
    parser.add_argument("--rev_row",
                        help="Whether to use the S-shaped alignment. Do NOT use this option when fitting an image using the --collage option",
                        action="store_true")
    parser.add_argument("--rev_sort",
                        help="Sort in the reverse direction. Do NOT use this option when fitting an image using the --collage option",
                        action="store_true")
    parser.add_argument("--dup", type=int, default=1, help="Duplicate the set of images by how many times")
    parser.add_argument("--ctype", type=str, default="float16",
                        help="Type of the cost matrix. Usually float16 is sufficient",
                        choices=["float16", "float32", "float64", "int16", "int32"])

    args = parser.parse_args()

    pic_path = args.path
    files = list(os.walk(pic_path))[0][-1]
    try:
        files.remove("cache.pkl")
    except ValueError:
        pass

    num_imgs = len(files)
    img_size = args.size, args.size
    rev = args.rev_row

    imgs = []
    for i, img_file in tqdm(enumerate(files), total=len(files), desc="[Reading images]", unit="imgs"):
        try:
            img = cv2.resize(cv2.imread(join(pic_path, img_file)), img_size, interpolation=cv2.INTER_CUBIC)
        except:
            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        imgs.append(img)

    if len(args.collage) == 0:
        # aspect ratio
        ratio = tuple(args.ratio)
        rw, rh = ratio
        possible_wh = []
        print("Calculating grid size...")
        for width in range(1, num_imgs):
            height = num_imgs // width
            possible_wh.append((width, height))

        best_wh = min(possible_wh, key=lambda x: ((x[0] / x[1]) - (rw / rh)) ** 2)

        print("Calculated grid size based on your aspect ratio:", best_wh)
        print("Note", num_imgs - best_wh[0] * best_wh[1], "of your friends will be thrown away from the collage")

        result_grid = best_wh

        print("Sorting images...")
        if args.sort.startswith("pca_"):
            sort_function = eval(args.sort)
            from sklearn.decomposition import PCA

            img_keys = PCA(1).fit_transform(np.array(list(map(sort_function, imgs))))[:, 0]
        elif args.sort.startswith("tsne_"):
            sort_function = eval(args.sort.replace("tsne", "pca"))
            from sklearn.manifold import TSNE

            img_keys = TSNE(n_components=1, n_iter=300).fit_transform(list(map(sort_function, imgs)))[:, 0]
        elif args.sort.startswith("umap_"):
            sort_function = eval(args.sort.replace("umap", "pca"))
            import umap

            img_keys = umap.UMAP(n_components=1, n_neighbors=15).fit_transform(list(map(sort_function, imgs)))[:, 0]
        elif args.sort == "none":
            img_keys = np.array(list(range(0, num_imgs)))
        else:
            sort_function = eval(args.sort)
            img_keys = list(map(sort_function, imgs))
            if type(img_keys[0]) == tuple:
                img_keys = list(map(lambda x: x[0], img_keys))
            img_keys = np.array(img_keys)

        sorted_imgs = np.array(imgs)[np.argsort(img_keys)]
        if args.rev_sort:
            sorted_imgs = list(reversed(sorted_imgs))
    else:
        assert isfile(args.collage)
        from scipy.spatial.distance import cdist

        collage = cv2.imread(args.collage)
        rh, rw, _ = collage.shape
        possible_wh = []

        print("Duplicating {} times".format(args.dup))
        imgs_copy = list(map(np.copy, imgs))
        for i in range(args.dup - 1):
            imgs.extend(imgs_copy)

        num_imgs = len(imgs)

        for width in range(1, num_imgs):
            height = num_imgs // width
            possible_wh.append((width, height))

        best_wh = min(possible_wh, key=lambda x: ((x[0] / x[1]) - (rw / rh)) ** 2)
        result_grid = best_wh

        print("Calculated grid size based on your aspect ratio:", best_wh)
        print("Note:", num_imgs - best_wh[0] * best_wh[1], "images will be thrown away from the collage")

        # it's VERY important to remove redundant images
        # this makes sure that cost_matrix is a square
        imgs = imgs[:best_wh[0] * best_wh[1]]


        def chl_mean_hsv(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])


        def chl_mean_bgr(img):
            return np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])


        def chl_mean_lab(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            return np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])


        def lumm(s):
            return [0.241 * s[0] + 0.691 * s[1] + 0.068 * s[2]]


        def av_lumm(img):
            return [np.mean(np.sqrt(0.241 * img[:, :, 0] + 0.691 * img[:, :, 1] + 0.068 * img[:, :, 2]))]


        print("Computing cost matrix...")
        if args.copt == "hsv":
            img_keys = np.array(list(map(chl_mean_hsv, imgs)))
            collage = cv2.cvtColor(collage, cv2.COLOR_BGR2HSV)
        elif args.copt == "bgr":
            img_keys = np.array(list(map(chl_mean_bgr, imgs)))
        elif args.copt == "lab":
            img_keys = np.array(list(map(chl_mean_lab, imgs)))
            collage = cv2.cvtColor(collage, cv2.COLOR_BGR2LAB)
        elif args.copt == "lum":
            img_keys = np.array(list(map(av_lumm, imgs)))

        else:
            raise Exception()

        resized_collage = cv2.resize(collage, best_wh, cv2.INTER_CUBIC).reshape(best_wh[0] * best_wh[1], 3)

        if args.copt == "lum":
            resized_collage = list(map(lumm, resized_collage))

        cost_matrix = cdist(img_keys, resized_collage, metric="euclidean")
        # print(np.max(cost_matrix), np.min(cost_matrix))
        ctype = eval("np." + args.ctype)
        cost_matrix = ctype(cost_matrix)

        print("Computing optimal pairing on a {}x{} matrix...".format(cost_matrix.shape[0], cost_matrix.shape[1]))
        try:
            from lap import lapjv

            _, _, cols = lapjv(cost_matrix)
        except ImportError:
            import lapsolver

            _, cols = lapsolver.solve_dense(cost_matrix)
            cols = np.argsort(cols)

        sorted_imgs = np.array(imgs)[cols]

    for i in tqdm(range(result_grid[1]), desc="[Merging]", unit="rows"):
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
