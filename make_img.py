import os
from os.path import *
from typing import Any, Union

import numpy as np
import cv2
import argparse
import random

from numpy.core.multiarray import ndarray
from tqdm import tqdm
import concurrent.futures as con


def bgr_chl_sum(img: np.ndarray) -> [float, float, float]:
    return np.sum(img[:, :, 0]), np.sum(img[:, :, 1]), np.sum(img[:, :, 2])


def bgr_sum(img: np.ndarray) -> float:
    return np.sum(img)


def av_hue(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0])


def av_sat(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 1])


def hsv(img: np.ndarray) -> [float, float, float]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])


def av_lum(img) -> float:
    return np.mean(np.sqrt(0.241 * img[:, :, 0] + 0.691 * img[:, :, 1] + 0.068 * img[:, :, 2]))


def pca_lum(img: np.ndarray) -> np.ndarray:
    return np.sqrt(0.241 * img[:, :, 0] + 0.691 * img[:, :, 1] + 0.068 * img[:, :, 2]).flatten()


def pca_sat(img: np.ndarray) -> np.ndarray:
    return (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]).flatten()


def pca_bgr(img: np.ndarray) -> np.ndarray:
    return img.flatten()


def pca_hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV).flatten()


def pca_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).flatten()


def pca_hue(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0].flatten()


def pca_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()


def rand(img: np.ndarray) -> float:
    return random.random()


def calculate_grid_size(rw: int, rh: int, num_imgs: int, v=False) -> tuple:
    possible_wh = []
    if v: print("Calculating grid size...")
    for width in range(1, num_imgs):
        height = num_imgs // width
        possible_wh.append((width, height))

    return min(possible_wh, key=lambda x: ((x[0] / x[1]) - (rw / rh)) ** 2)


def make_collage(grid: tuple, sorted_imgs: list, rev=False, v=False) -> np.ndarray:
    if v: print("Mering images...")
    for i in range(grid[1]):
        if rev and i % 2 == 1:
            img_f = sorted_imgs[(i + 1) * grid[0] - 1]
            for j in range(grid[0] - 2, -1, -1):
                img_t = sorted_imgs[i * grid[0] + j]
                img_f = np.append(img_f, img_t, axis=1)
            if i == 0:
                combined_img = img_f
            else:
                combined_img = np.append(combined_img, img_f, axis=0)
        else:
            img_f = sorted_imgs[i * grid[0]]
            for j in range(1, grid[0]):
                img_t = sorted_imgs[i * grid[0] + j]
                img_f = np.append(img_f, img_t, axis=1)
            if i == 0:
                combined_img = img_f
            else:
                combined_img = np.append(combined_img, img_f, axis=0)
    return combined_img


def calculate_decay_weights_normal(shape: tuple, sigma: float = 1) -> np.ndarray:
    assert sigma != 0
    from scipy.stats import truncnorm
    h, w = shape
    h_arr = truncnorm.pdf(np.linspace(-1, 1, h), -1, 1, loc=0, scale=abs(sigma))
    w_arr = truncnorm.pdf(np.linspace(-1, 1, w), -1, 1, loc=0, scale=abs(sigma))
    h_arr, w_arr = np.meshgrid(h_arr, w_arr)
    if sigma > 0:
        weights = h_arr * w_arr
    else:
        weights = 1 - h_arr * w_arr
    return weights


def sort_collage(imgs: list, ratio: tuple, sort_method="pca_lab", rev_sort=False, v=False) -> [tuple, np.ndarray]:
    num_imgs = len(imgs)
    result_grid = calculate_grid_size(ratio[0], ratio[1], num_imgs, v)

    if v: print("Calculated grid size based on your aspect ratio:", result_grid)
    if v: print("Note", num_imgs - result_grid[0] * result_grid[1],
                "of your friends will be thrown away from the collage")

    if v: print("Sorting images...")
    if sort_method.startswith("pca_"):
        sort_function = eval(sort_method)
        from sklearn.decomposition import PCA

        img_keys = PCA(1).fit_transform(np.array(list(map(sort_function, imgs))))[:, 0]
    elif sort_method.startswith("tsne_"):
        sort_function = eval(sort_method.replace("tsne", "pca"))
        from sklearn.manifold import TSNE

        img_keys = TSNE(n_components=1, n_iter=300).fit_transform(list(map(sort_function, imgs)))[:, 0]
    elif sort_method.startswith("umap_"):
        sort_function = eval(sort_method.replace("umap", "pca"))
        import umap

        img_keys = umap.UMAP(n_components=1, n_neighbors=15).fit_transform(list(map(sort_function, imgs)))[:, 0]
    elif sort_method == "none":
        img_keys = np.array(list(range(0, num_imgs)))
    else:
        sort_function = eval(sort_method)
        img_keys = list(map(sort_function, imgs))
        if type(img_keys[0]) == tuple:
            img_keys = list(map(lambda x: x[0], img_keys))
        img_keys = np.array(img_keys)

    sorted_imgs = np.array(imgs)[np.argsort(img_keys)]
    if rev_sort:
        sorted_imgs = list(reversed(sorted_imgs))

    return result_grid, sorted_imgs


def calculate_collage(collage_file: str, imgs: list, dup: int = 1, copt="lab", ctype="float16", sigma: float = 1,
                      v=False) -> [tuple, list, float]:
    assert isfile(collage_file)
    from scipy.spatial.distance import cdist
    from lap import lapjv

    collage = cv2.imread(collage_file)
    rh, rw, _ = collage.shape

    if v: print("Duplicating {} times".format(dup))

    imgs = list(map(np.copy, imgs))
    imgs_copy = list(map(np.copy, imgs))
    for i in range(dup - 1):
        imgs.extend(imgs_copy)

    num_imgs = len(imgs)

    result_grid = calculate_grid_size(rw, rh, num_imgs, v)

    if v: print("Calculated grid size based on your aspect ratio:", result_grid)
    if v: print("Note:", num_imgs - result_grid[0] * result_grid[1], "images will be thrown away from the collage")

    # it's VERY important to remove redundant images
    # this makes sure that cost_matrix is a square
    imgs = imgs[:result_grid[0] * result_grid[1]]

    def chl_mean_hsv(weights):
        ratio = weights.shape[0] * weights.shape[1] / np.sum(weights)

        def f(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return ratio * np.average(img[:, :, 0], weights=weights), \
                   ratio * np.average(img[:, :, 1], weights=weights), \
                   ratio * np.average(img[:, :, 2], weights=weights)

        return f

    def chl_mean_bgr(weights):
        ratio = weights.shape[0] * weights.shape[1] / np.sum(weights)

        def f(img):
            return ratio * np.average(img[:, :, 0], weights=weights), \
                   ratio * np.average(img[:, :, 1], weights=weights), \
                   ratio * np.average(img[:, :, 2], weights=weights)

        return f

    def chl_mean_lab(weights):
        ratio = weights.shape[0] * weights.shape[1] / np.sum(weights)

        def f(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            return ratio * np.average(img[:, :, 0], weights=weights), \
                   ratio * np.average(img[:, :, 1], weights=weights), \
                   ratio * np.average(img[:, :, 2], weights=weights)

        return f

    weights = calculate_decay_weights_normal(imgs[0].shape[:2], sigma)

    if v: print("Computing cost matrix...")
    if copt == "hsv":
        img_keys = np.array(list(map(chl_mean_hsv(weights), imgs)))
        collage = cv2.cvtColor(collage, cv2.COLOR_BGR2HSV)
    elif copt == "bgr":
        img_keys = np.array(list(map(chl_mean_bgr(weights), imgs)))
    elif copt == "lab":
        img_keys = np.array(list(map(chl_mean_lab(weights), imgs)))
        collage = cv2.cvtColor(collage, cv2.COLOR_BGR2LAB)
    else:
        raise Exception()

    resized_collage = cv2.resize(collage, result_grid, cv2.INTER_CUBIC).reshape(result_grid[0] * result_grid[1], 3)

    cost_matrix = cdist(img_keys, resized_collage, metric="euclidean")

    ctype = eval("np." + ctype)
    cost_matrix = ctype(cost_matrix)

    if v: print("Computing optimal assignment on a {}x{} matrix...".format(cost_matrix.shape[0], cost_matrix.shape[1]))

    cost, _, cols = lapjv(cost_matrix)
    if v: print("Total assignment cost:", cost)

    return result_grid, np.array(imgs)[cols], cost


def save_img(img: np.ndarray, path: str, suffix: str, v=False) -> None:
    if len(path) == 0:
        path = 'result_{}.png'.format(suffix)
        if v: print("Saving to", path)
        cv2.imwrite(path, img)
    else:
        if len(suffix) == 0:
            if v: print("Saving to", path)
            cv2.imwrite(path, img)
        else:
            path = path.split(".")
            path = path[0] + "_{}".format(suffix) + "." + path[1]
            if v: print("Saving to", path)
            cv2.imwrite(path, img)


def ss_wrapper(a):
    return sort_collage(*a)


def sc_wrapper(a):
    return calculate_collage(*a)


if __name__ == "__main__":
    all_sort_methods = ["none", "bgr_sum", "av_hue", "av_sat", "av_lum", "rand",
                        "pca_bgr", "pca_hsv", "pca_lab", "pca_gray", "pca_lum", "pca_sat", "pca_hue",
                        "tsne_bgr", "tsne_hsv", "tsne_lab", "tsne_gray", "tsne_lum", "tsne_sat", "tsne_hue",
                        "umap_bgr", "umap_hsv", "umap_lab", "umap_gray", "umap_lum", "umap_sat", "umap_hue",
                        ]
    all_color_spaces = ["hsv", "bgr", "lab"]
    all_sigmas = np.concatenate((np.arange(-1, -0.45, 0.05), np.arange(0.5, 1.05, 0.05)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the downloaded head images",
                        default=join(dirname(__file__), "img"), type=str)
    parser.add_argument("--out", help="The name of the output image", default="", type=str)
    parser.add_argument("--size", help="Size of each image in pixels", type=int, default=100)
    parser.add_argument("--ratio", help="Aspect ratio", nargs=2, type=int, default=(16, 9))
    parser.add_argument("--sort", help="Sort methods", choices=all_sort_methods,
                        type=str, default="bgr_sum")
    parser.add_argument("--collage", type=str, default="",
                        help="If you want to fit an image, specify the image path here")
    parser.add_argument("--copt", type=str, default="lab", choices=all_color_spaces,
                        help="Methods to use when fitting an image")
    parser.add_argument("--rev_row",
                        help="Whether to use the S-shaped alignment. "
                             "Do NOT use this option when fitting an image using the --collage option",
                        action="store_true")
    parser.add_argument("--rev_sort",
                        help="Sort in the reverse direction. "
                             "Do NOT use this option when fitting an image using the --collage option",
                        action="store_true")
    parser.add_argument("--dup", type=int, default=1, help="Duplicate the set of images by how many times")
    parser.add_argument("--ctype", type=str, default="float16",
                        help="Type of the cost matrix. Usually float16 is sufficient",
                        choices=["float16", "float32", "float64", "int16", "int32"])
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--exp", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    v = args.verbose

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
            imgs.append(cv2.resize(cv2.imread(join(pic_path, img_file)), img_size, interpolation=cv2.INTER_CUBIC))
        except:
            imgs.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8))

    if len(args.collage) == 0:
        if args.exp:

            pool = con.ProcessPoolExecutor(8)
            futures = []


            def callback(x, out, rev_row, suffix, pbar, v=False):
                result_grid, sorted_imgs = x.result()
                combined_img = make_collage(result_grid, sorted_imgs, rev_row, v)
                save_img(combined_img, out, suffix, v)
                pbar.update()


            pbar = tqdm(total=len(all_sort_methods), desc="[Experimenting]", unit="exps")
            for sort_method in all_sort_methods:
                f = pool.submit(ss_wrapper, (imgs, args.ratio, sort_method, args.rev_sort, v))
                (lambda sort_method: f.add_done_callback(
                    lambda x: callback(x, args.out, args.rev_row, sort_method, pbar, v)))(sort_method)
                futures.append(f)

            exit(0)

        else:
            result_grid, sorted_imgs = sort_collage(imgs, args.ratio, args.sort, args.rev_sort, v)
            save_img(make_collage(result_grid, sorted_imgs, args.rev_row, v), args.out, args.sort, v)
    else:
        if args.exp:
            import matplotlib.pyplot as plt

            pool = con.ProcessPoolExecutor(6)
            futures = []


            def callback(x, out, rev_row, suffix, pbar, v=False):
                result_grid, sorted_imgs = x.result()
                combined_img = make_collage(result_grid, sorted_imgs, rev_row, v)
                save_img(combined_img, out, suffix, v)
                pbar.update()


            total_steps = len(all_sigmas) * len(all_color_spaces)
            for sigma in all_sigmas:
                for color_space in all_color_spaces:
                    f = pool.submit(sc_wrapper, (args.collage, imgs, args.dup, color_space,
                                                 args.ctype, sigma, v))
                    futures.append((f, sigma, color_space))

            cost_vis = np.zeros((len(all_sigmas), len(all_color_spaces)))
            r = 0
            c = 0
            for (f, sigma, color_space) in tqdm(futures, desc="[Experimenting]", unit="exps"):
                suffix = "{}_{}".format(color_space, np.round(sigma, 2))
                result_grid, sorted_imgs, cost = f.result()
                save_img(make_collage(result_grid, sorted_imgs, args.rev_row, v), args.out, suffix, v)
                cost_vis[r, c] = cost
                c += 1
                if c % len(all_color_spaces) == 0:
                    r += 1
                    c = 0

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            X, Y = np.meshgrid(np.arange(len(all_color_spaces)), all_sigmas)
            surf = ax.plot_surface(X, Y, cost_vis)
            ax.set_xlabel("Color Space")
            ax.set_ylabel("Sigma")
            ax.set_zlabel("Cost")
            plt.xticks(np.arange(len(all_color_spaces)), all_color_spaces)
            plt.show()
            exit(0)

        else:
            result_grid, sorted_imgs, cost = calculate_collage(args.collage, imgs, args.dup,
                                                               args.copt, args.ctype, args.sigma, v)
            save_img(make_collage(result_grid, sorted_imgs, args.rev_row, v),
                     args.out, "{}_{}".format(args.copt, np.round(args.sigma, 2)), v)
