import os
import argparse
import random
import concurrent.futures as con
import multiprocessing as mp
import sys
import time
import platform
from typing import List, Tuple, Callable
import traceback
from math import ceil

import cv2
import numpy as np
from tqdm import tqdm
from scipy.stats import truncnorm
from scipy.spatial.distance import cdist


if mp.current_process().name != "MainProcess":
    sys.stdout = open(os.devnull, "w")
    sys.stderr = sys.stdout

pbar_ncols = None


def bgr_chl_sum(img: np.ndarray) -> Tuple[float, float, float]:
    """
    compute the channel-wise sum of the BGR color space
    """
    return np.sum(img[:, :, 0]), np.sum(img[:, :, 1]), np.sum(img[:, :, 2])


def bgr_sum(img: np.ndarray) -> float:
    """
    compute the sum of all RGB values across an image
    """
    return np.sum(img)


def av_hue(img: np.ndarray) -> float:
    """
    compute the average hue of all pixels in HSV color space
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0])


def av_sat(img: np.ndarray) -> float:
    """
    compute the average saturation of all pixels in HSV color space
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 1])


def hsv(img: np.ndarray) -> Tuple[float, float, float]:
    """
    compute the channel-wise average of the image in HSV color space
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])


def av_lum(img) -> float:
    """
    compute the average luminosity
    """
    return np.mean(np.sqrt(0.241 * img[:, :, 0] + 0.691 * img[:, :, 1] + 0.068 * img[:, :, 2]))


def pca_lum(img: np.ndarray) -> np.ndarray:
    """
    flatten the image using the luminosity of each pixel
    """
    return np.sqrt(0.241 * img[:, :, 0] + 0.691 * img[:, :, 1] + 0.068 * img[:, :, 2]).flatten()


def pca_sat(img: np.ndarray) -> np.ndarray:
    """
    flatten the image using the saturation of each pixel
    """
    return (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]).flatten()


def pca_bgr(img: np.ndarray) -> np.ndarray:
    """
    flatten the image in BGR color space
    """
    return img.flatten()


def pca_hsv(img: np.ndarray) -> np.ndarray:
    """
    flatten the image in HSV color space
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV).flatten()


def pca_lab(img: np.ndarray) -> np.ndarray:
    """
    flatten the image in LAB color space
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab).flatten()


def pca_hue(img: np.ndarray) -> np.ndarray:
    """
    flatten the image using the hue value of each pixel
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0].flatten()


def pca_gray(img: np.ndarray) -> np.ndarray:
    """
    flatten the image in gray scale
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()


def rand(img: np.ndarray) -> float:
    """
    generate a random number for each image
    """
    return random.random()


class JVOutWrapper:
    """
    The output wrapper for displaying the progress of J-V algorithm
    Only available for the GUI running in Linux or Unix-like systems
    """

    def __init__(self, io_wrapper):
        self.io_wrapper = io_wrapper
        self.tqdm = None

    def write(self, lines: str):
        for line in lines.split("\n"):
            if not line.startswith("lapjv: "):
                continue
            if line.find("AUGMENT SOLUTION row ") > 0:
                line = line.replace(" ", "")
                slash_idx = line.find("/")
                s_idx = line.find("[")
                e_idx = line.find("]")
                if s_idx > -1 and slash_idx > -1 and e_idx > -1:
                    if self.tqdm:
                        self.tqdm.n = int(line[s_idx + 1:slash_idx])
                        self.tqdm.update(0)
                    else:
                        self.tqdm = tqdm(file=self.io_wrapper, ncols=self.io_wrapper.width,
                                         total=int(line[slash_idx + 1:e_idx]))
                continue
            if not self.tqdm:
                self.io_wrapper.write(line + "\n")

    def flush(self):
        pass

    def finish(self):
        if self.tqdm:
            self.tqdm.n = self.tqdm.total
            self.tqdm.update(0)
            self.tqdm.close()


def calc_grid_size(rw: int, rh: int, num_imgs: int) -> Tuple[int, int]:
    """
    :param rw: the width of the target image
    :param rh: the height of the target image
    :param num_imgs: number of images available
    :return: an optimal grid size
    """
    possible_wh = []
    for width in range(1, num_imgs):
        height = num_imgs // width
        possible_wh.append((width, height))

    return min(possible_wh, key=lambda x: ((x[0] / x[1]) - (rw / rh)) ** 2)


def make_collage(grid: Tuple[int, int], sorted_imgs: List[np.ndarray], rev: bool = False) -> np.ndarray:
    """
    :param grid: grid size
    :param sorted_imgs: list of images sorted in correct position
    :param rev: whether to have opposite alignment for consecutive rows
    :return: a collage
    """
    size, _, _ = sorted_imgs[0].shape
    print("Aligning images on the grid...")
    combined_img = np.zeros((grid[1] * size, grid[0] * size, 3), np.uint8)
    for i in range(grid[1]):
        if rev and i % 2 == 1:
            for j in range(0, grid[0]):
                combined_img[i * size:(i + 1) * size, j * size:(j + 1)
                             * size, :] = sorted_imgs[i * grid[0] + grid[0] - j - 1]
        else:
            for j in range(0, grid[0]):
                combined_img[i * size:(i + 1) * size, j * size:(j + 1)
                             * size, :] = sorted_imgs[i * grid[0] + j]
    return combined_img


def calc_decay_weights_normal(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """
    Generate a matrix of probabilities (as weights) sampled from a truncated bivariate normal distribution
    centered at (0, 0) whose covariance matrix is equal to sigma times the identity matrix

    Basically it's a product of two identical and independent normal distributions truncated in the range [-1,1]
    with mean=0 and sigma=sigma

    Generate an "upside-down" bivariate normal if sigma is negative.

    Use sigma=0 to generate a matrix with uniform weights
    """
    h, w = shape
    if sigma == 0:
        return np.ones((h, w))
    else:
        h_arr = truncnorm.pdf(np.linspace(-1, 1, h), -1,
                            1, loc=0, scale=abs(sigma))
        w_arr = truncnorm.pdf(np.linspace(-1, 1, w), -1,
                            1, loc=0, scale=abs(sigma))
        h_arr, w_arr = np.meshgrid(h_arr, w_arr)
        if sigma > 0:
            weights = h_arr * w_arr
        else:
            weights = 1 - h_arr * w_arr
        
        # if the sum of weights is too small, return a matrix with uniform weights
        if abs(np.sum(weights)) > 0.00001:
            return weights
        else:
            return np.ones((h, w))


def sort_collage(imgs: List[np.ndarray], ratio: Tuple[int, int], sort_method="pca_lab", 
                 rev_sort=False) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    :param imgs: list of images
    :param ratio: The aspect ratio of the collage
    :param sort_method:
    :param rev_sort: whether to reverse the sorted array
    :return: calculated grid size and the sorted image array
    """
    num_imgs = len(imgs)
    grid = calc_grid_size(ratio[0], ratio[1], num_imgs)

    print("Calculated grid size based on your aspect ratio:", grid)
    print("Note that", num_imgs - grid[0] * grid[1],
          "images will be thrown away from the collage")
    print("Sorting images...")
    t = time.time()
    if sort_method.startswith("pca_"):
        sort_function = eval(sort_method)
        from sklearn.decomposition import PCA

        img_keys = PCA(1).fit_transform(list(map(sort_function, imgs)))[:, 0]
    elif sort_method.startswith("tsne_"):
        sort_function = eval(sort_method.replace("tsne", "pca"))
        from sklearn.manifold import TSNE

        img_keys = TSNE(n_components=1, verbose=1, init="pca").fit_transform(
            list(map(sort_function, imgs)))[:, 0]
    elif sort_method.startswith("umap_"):
        sort_function = eval(sort_method.replace("umap", "pca"))

        import umap

        img_keys = umap.UMAP(n_components=1, verbose=1).fit_transform(
            list(map(sort_function, imgs)))[:, 0]
    elif sort_method == "none":
        img_keys = np.array(list(range(0, num_imgs)))
    else:
        sort_function = eval(sort_method)
        img_keys = list(map(sort_function, imgs))

        # only take the first value if img_keys is a list of tuples
        if isinstance(img_keys[0], tuple):
            img_keys = list(map(lambda x: x[0], img_keys))
        img_keys = np.array(img_keys)

    sorted_imgs = np.array(imgs)[np.argsort(img_keys)]
    if rev_sort:
        sorted_imgs = list(reversed(sorted_imgs))

    print("Time taken: {}s".format(np.round(time.time() - t, 2)))

    return grid, sorted_imgs


def chl_mean_hsv(weights: np.ndarray) -> Callable:
    """
    return a function that can calculate the channel-wise average 
    of the input picture in HSV color space
    """
    def f(img: np.ndarray) -> Tuple[float, float, float]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.average(img[:, :, 0], weights=weights), \
            np.average(img[:, :, 1], weights=weights), \
            np.average(img[:, :, 2], weights=weights)

    return f


def chl_mean_hsl(weights: np.ndarray) -> Callable:
    """
    return a function that can calculate the channel-wise average 
    of the input picture in HSL color space
    """
    def f(img: np.ndarray) -> Tuple[float, float, float]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return np.average(img[:, :, 0], weights=weights), \
            np.average(img[:, :, 1], weights=weights), \
            np.average(img[:, :, 2], weights=weights)

    return f


def chl_mean_bgr(weights: np.ndarray) -> Callable:
    """
    return a function that can calculate the channel-wise average 
    of the input picture in BGR color space
    """
    def f(img: np.ndarray) -> Tuple[float, float, float]:
        return np.average(img[:, :, 0], weights=weights), \
            np.average(img[:, :, 1], weights=weights), \
            np.average(img[:, :, 2], weights=weights)

    return f


def chl_mean_lab(weights: np.ndarray) -> Callable:
    """
    return a function that can calculate the channel-wise average 
    of the input picture in LAB color space
    """
    def f(img: np.ndarray) -> Tuple[float, float, float]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return np.average(img[:, :, 0], weights=weights), \
            np.average(img[:, :, 1], weights=weights), \
            np.average(img[:, :, 2], weights=weights)

    return f


def calc_saliency_map(dest_img: np.ndarray, lower_thresh: int = 50, fill_bg: bool = False, 
                      bg_color: Tuple[int,int,int] = (255, 255, 255)) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    :param dest_img: the destination image
    :param lower_thresh: lower threshold for salient object detection. 
                         If it's -1, then the threshold value will be adaptive
    :param fill_bg: whether to fill the background with the color specified
    :param bg_color: background color
    :return: [the copy of the destination image with background filled (if fill_bg), 
            threshold map, object area in pixels]
    """

    flag = cv2.THRESH_BINARY
    if lower_thresh == -1:
        lower_thresh = 0
        flag = flag | cv2.THRESH_OTSU
    
    dest_img = np.copy(dest_img)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency.computeSaliency(dest_img)
    saliency_map = (saliency_map * 255).astype(np.uint8)
    _, thresh = cv2.threshold(saliency_map, lower_thresh, 255, flag)
    thresh = thresh.astype(np.uint8)
    
    h, w, _ = dest_img.shape
    obj_area = 0

    if fill_bg:
        for i in range(h):
            for j in range(w):
                if thresh[i, j] != 0:
                    obj_area += 1
                else:
                    dest_img[i, j, :] = np.array(bg_color, np.uint8)
    else:
        obj_area = np.count_nonzero(thresh)

    return dest_img, thresh, obj_area


def calc_salient_col_even_fast(dest_img_path: str, imgs: List[np.ndarray], dup: int = 1,
                               colorspace: str = "lab", ctype: str = "float16", sigma: float = 1.0,
                               metric: str = "euclidean", lower_thresh: int = 50,
                               background: Tuple[int, int, int] = (255, 255, 255),
                               v=None) -> Tuple[Tuple[int, int], List[np.ndarray], float]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels constitute of salient objects of the
    target image, with the restriction that every image should be used the same amount of times

    :param dest_img_path: path to the destination image file
    :param imgs: list of images
    :param dup: number of times to duplicate the set of images
    :param colorspace: name of the colorspace used
    :param ctype: ctype of the cost matrix
    :param sigma:
    :param v: verbose
    :param lower_thresh: lower threshold for object detection
    :param background: background color
    :return: [gird size, sorted images, total assignment cost]
    """
    assert os.path.isfile(dest_img_path)
    
    t = time.time()
    print("Duplicating {} times".format(dup))

    # avoid modifying the original array
    imgs = list(map(np.copy, imgs))
    imgs_copy = list(map(np.copy, imgs))
    for i in range(dup - 1):
        imgs.extend(imgs_copy)

    dest_img = imread(dest_img_path)
    rh, rw, _ = dest_img.shape

    threshold = lower_thresh
    _, threshed_map, obj_area = calc_saliency_map(dest_img, threshold)

    dest_img_copy = np.copy(dest_img)    
    grid = (0, 0)
    pbar = tqdm(unit=" iteration", desc="[Computing saliency & grid]", ncols=pbar_ncols)
    while True:
        num_imgs = round(rh * rw / obj_area * len(imgs))

        grid = calc_grid_size(rw, rh, num_imgs)

        dest_img = cv2.resize(dest_img_copy, grid, cv2.INTER_AREA)
        rh, rw, _ = dest_img.shape

        _, threshed_map, obj_area = calc_saliency_map(dest_img, threshold)
        diff = len(imgs) - obj_area

        if threshold != -1:
            if diff > 0:
                threshold -= 3
                if threshold < 1:
                    threshold = 1
            else:
                threshold += 3
                if threshold > 254:
                    threshold = 254

        pbar.update(1)
        if diff >= 0 and diff < ceil(len(imgs) / dup / 3) or pbar.n > 50:
            break

    pbar.close()
    if threshold != -1:
        print("Final threshold:", threshold)

    print("Grid size {} calculated in {} iterations".format(grid, pbar.n))

    dest_obj = []
    coor = []
    for i in range(rh):
        for j in range(rw):
            if threshed_map[i, j] != 0:
                dest_obj.append(dest_img[i, j, :])
                coor.append(i * rw + j)
            else:
                dest_img[i, j, :] = np.array([background[2], background[1], background[0]], np.uint8)

    if len(imgs) > len(dest_obj):
        print("Note:", len(imgs) - len(dest_obj),
              "images will be thrown away from the collage")
    imgs = imgs[:len(dest_obj)]

    print("Computing cost matrix...")

    dest_obj = np.array([dest_obj], np.uint8)
    weights = calc_decay_weights_normal(imgs[0].shape[:2], sigma)
    if colorspace == "hsv":
        img_keys = np.array(list(map(chl_mean_hsv(weights), imgs)))
        dest_obj = cv2.cvtColor(dest_obj, cv2.COLOR_BGR2HSV)
    elif colorspace == "hsl":
        img_keys = np.array(list(map(chl_mean_hsl(weights), imgs)))
        dest_obj = cv2.cvtColor(dest_obj, cv2.COLOR_BGR2HLS)
    elif colorspace == "bgr":
        img_keys = np.array(list(map(chl_mean_bgr(weights), imgs)))
    elif colorspace == "lab":
        img_keys = np.array(list(map(chl_mean_lab(weights), imgs)))
        dest_obj = cv2.cvtColor(dest_obj, cv2.COLOR_BGR2Lab)

    dest_obj = dest_obj[0]

    cost_matrix = cdist(img_keys, dest_obj, metric=metric)

    np_ctype = eval("np." + ctype)
    cost_matrix = np_ctype(cost_matrix)

    print("Computing optimal assignment on a {}x{} matrix...".format(
        cost_matrix.shape[0], cost_matrix.shape[1]))

    from lapjv import lapjv

    if v is not None and (platform.system() == "Linux" or platform.system() == "Darwin") and v.gui:
        try:
            from wurlitzer import pipes, STDOUT
            from wurlitzer import Wurlitzer
            Wurlitzer.flush_interval = 0.1
            wrapper = JVOutWrapper(v)
            with pipes(stdout=wrapper, stderr=STDOUT):
                _, cols, cost = lapjv(cost_matrix, verbose=1)
                wrapper.finish()
        except ImportError:
            _, cols, cost = lapjv(cost_matrix)

    else:
        _, cols, cost = lapjv(cost_matrix)

    cost = cost[0]

    del cost_matrix

    paired = np.array(imgs)[cols]

    white = np.ones(imgs[0].shape, np.uint8)
    white[:, :, :] = [background[2], background[1], background[0]]

    filled = []
    counter = 0
    for i in range(grid[0] * grid[1]):
        if i in coor:
            filled.append(paired[counter])
            counter += 1
        else:
            filled.append(white)

    print("Total assignment cost:", cost)
    print("Time taken: {}s".format((np.round(time.time() - t, 2))))

    return grid, filled, cost


def calc_col_even(dest_img_path: str, imgs: List[np.ndarray], dup: int = 1,
                  colorspace: str = "lab", ctype: str = "float16", sigma: float = 1.0,
                  metric: str = "euclidean", v=None) -> Tuple[Tuple[int, int], List[np.ndarray], float]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels of the target image,
    with the restriction that every image should be used the same amount of times

    :param dest_img_path: path to the destination image file
    :param imgs: list of images
    :param dup: number of times to duplicate the set of images
    :param colorspace: name of the colorspace used
    :param ctype: ctype of the cost matrix
    :param sigma:
    :param v: verbose
    :return: [gird size, sorted images, total assignment cost]
    """
    assert os.path.isfile(dest_img_path)

    print("Duplicating {} times".format(dup))

    # avoid modifying the original array
    imgs = list(map(np.copy, imgs))
    imgs_copy = list(map(np.copy, imgs))
    for i in range(dup - 1):
        imgs.extend(imgs_copy)

    num_imgs = len(imgs)

    # Compute the grid size based on the number images that we have
    dest_img = imread(dest_img_path)
    rh, rw, _ = dest_img.shape
    grid = calc_grid_size(rw, rh, num_imgs)

    print("Calculated grid size based on the aspect ratio of the image provided:",
          grid)
    print("Note:", num_imgs - grid[0] * grid[1],
          "images will be thrown away from the collage")

    # it's VERY important to remove redundant images
    # this makes sure that cost_matrix is a square
    imgs = imgs[:grid[0] * grid[1]]

    # Resize the destination image so that it has the same size as the grid
    # This makes sure that each image in the list of images corresponds to a pixel of the destination image
    dest_img = cv2.resize(dest_img, grid, cv2.INTER_AREA)

    weights = calc_decay_weights_normal(imgs[0].shape[:2], sigma)
    t = time.time()
    print("Computing cost matrix...")
    if colorspace == "hsv":
        img_keys = np.array(list(map(chl_mean_hsv(weights), imgs)))
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2HSV)
    elif colorspace == "hsl":
        img_keys = np.array(list(map(chl_mean_hsl(weights), imgs)))
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2HLS)
    elif colorspace == "bgr":
        img_keys = np.array(list(map(chl_mean_bgr(weights), imgs)))
    elif colorspace == "lab":
        img_keys = np.array(list(map(chl_mean_lab(weights), imgs)))
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2Lab)

    dest_img = dest_img.reshape(grid[0] * grid[1], 3)

    # compute pair-wise distances
    cost_matrix = cdist(img_keys, dest_img, metric=metric)

    np_ctype = eval("np." + ctype)
    cost_matrix = np_ctype(cost_matrix)

    print("Computing optimal assignment on a {}x{} matrix...".format(
        cost_matrix.shape[0], cost_matrix.shape[1]))

    from lapjv import lapjv

    if v is not None and (platform.system() == "Linux" or platform.system() == "Darwin") and v.gui:
        try:
            from wurlitzer import pipes, STDOUT
            from wurlitzer import Wurlitzer
            Wurlitzer.flush_interval = 0.1
            wrapper = JVOutWrapper(v)
            with pipes(stdout=wrapper, stderr=STDOUT):
                _, cols, cost = lapjv(cost_matrix, verbose=1)
                wrapper.finish()
        except ImportError:
            _, cols, cost = lapjv(cost_matrix)

    else:
        _, cols, cost = lapjv(cost_matrix)

    cost = cost[0]

    print("Total assignment cost:", cost)
    print("Time taken: {}s".format((np.round(time.time() - t, 2))))

    # sometimes the cost matrix may be extremely large
    # manually delete it to free memory
    del cost_matrix

    return grid, np.array(imgs)[cols], cost


def calc_salient_col_dup(dest_img_path: str, imgs: List[np.ndarray], max_width: int = 80,
                         color_space="lab", sigma: float = 1.0, metric: str = "euclidean", lower_thresh: int = 127,
                         background: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[Tuple[int, int], List[np.ndarray], float]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels that constitute 
    of the salient objects of the target image, given that every image could be used arbitrary amount of times

    :param dest_img_path: path to the dest_img file
    :param imgs: list of images
    :param max_width: max_width of the resulting dest_img
    :param color_space: color space used
    :param sigma:
    :param lower_thresh: threshold for object detection
    :background background color
    :return: [gird size, sorted images, total assignment cost]
    """
    assert os.path.isfile(dest_img_path)

    t = time.time()

    # deep copy
    imgs = list(map(np.copy, imgs))
    dest_img = imread(dest_img_path)

    rh, rw, _ = dest_img.shape

    rh = round(rh * max_width / rw)
    grid = (max_width, rh)

    print("Calculated grid size based on the aspect ratio of the image provided:", grid)

    weights = calc_decay_weights_normal(imgs[0].shape[:2], sigma)
    dest_img = cv2.resize(dest_img, grid, cv2.INTER_AREA)

    dest_img, _, _ = calc_saliency_map(dest_img, lower_thresh, True, [background[2],background[1],background[0]])

    white = np.ones(imgs[0].shape, np.uint8) * 255
    white[:, :, :] = [background[2], background[1], background[0]]
    imgs.append(white)

    print("Computing costs...")
    if color_space == "hsv":
        img_keys = np.array(list(map(chl_mean_hsv(weights), imgs)))
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2HSV)
    elif color_space == "hsl":
        img_keys = np.array(list(map(chl_mean_hsl(weights), imgs)))
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2HLS)
    elif color_space == "bgr":
        img_keys = np.array(list(map(chl_mean_bgr(weights), imgs)))
    elif color_space == "lab":
        img_keys = np.array(list(map(chl_mean_lab(weights), imgs)))
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2Lab)

    dest_img = dest_img.reshape(grid[0] * grid[1], 3)
    sorted_imgs = []
    cost = 0
    for pixel in tqdm(dest_img, desc="[Computing assignments]", unit="pixel", unit_divisor=1000, unit_scale=True,
                      ncols=pbar_ncols):
        # Compute the distance between the current pixel and each image in the set
        dist = cdist(img_keys, np.array([pixel]), metric=metric)[:, 0]

        # Find the index of the image which best approximates the current pixel
        idx = np.argmin(dist)

        # Store that image
        sorted_imgs.append(imgs[idx])

        # Accumulate the distance to get the total cot
        cost += dist[idx]

    print("Time taken: {}s".format(np.round(time.time() - t, 2)))

    return grid, sorted_imgs, cost


def calc_col_dup(dest_img_path: str, imgs: list, max_width: int = 80, color_space="lab",
                 sigma: float = 1.0, metric: str = "euclidean") -> Tuple[Tuple[int, int], List[np.ndarray], float]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels of the target image,
    given that every image could be used arbitrary amount of times

    :param dest_img_path: path to the dest_img file
    :param imgs: list of images
    :param max_width: max_width of the resulting dest_img
    :param color_space: color space used
    :param sigma:
    :return: [gird size, sorted images, total assignment cost]
    """
    assert os.path.isfile(dest_img_path)

    dest_img = imread(dest_img_path)

    # Because we don't have a fixed total amount of images as we can used a single image
    # for arbitrary amount of times, we need user to specify the maximum width in order to determine the grid size.
    rh, rw, _ = dest_img.shape
    rh = round(rh * max_width / rw)
    grid = (max_width, rh)

    print("Calculated grid size based on the aspect ratio of the image provided:", grid)

    weights = calc_decay_weights_normal(imgs[0].shape[:2], sigma)
    dest_img = cv2.resize(dest_img, grid, cv2.INTER_AREA)
    t = time.time()
    print("Computing costs")
    if color_space == "hsv":
        img_keys = np.array(list(map(chl_mean_hsv(weights), imgs)))
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2HSV)
    elif color_space == "hsl":
        img_keys = np.array(list(map(chl_mean_hsl(weights), imgs)))
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2HLS)
    elif color_space == "bgr":
        img_keys = np.array(list(map(chl_mean_bgr(weights), imgs)))
    elif color_space == "lab":
        img_keys = np.array(list(map(chl_mean_lab(weights), imgs)))
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2Lab)

    dest_img = dest_img.reshape(grid[0] * grid[1], 3)

    sorted_imgs = []
    cost = 0
    for pixel in tqdm(dest_img, desc="[Computing assignments]", unit="pixel", unit_divisor=1000, unit_scale=True,
                      ncols=pbar_ncols):
        # Compute the distance between the current pixel and each image in the set
        dist = cdist(img_keys, np.array([pixel]), metric=metric)[:, 0]

        # Find the index of the image which best approximates the current pixel
        idx = np.argmin(dist)

        # Store that image
        sorted_imgs.append(imgs[idx])

        # Accumulate the distance to get the total cot
        cost += dist[idx]

    print("Time taken: {}s".format(np.round(time.time() - t, 2)))

    return grid, sorted_imgs, cost


def imwrite(filename: str, img: np.ndarray) -> None:
    ext = os.path.splitext(filename)[1]
    result, n = cv2.imencode(ext, img)

    if result:
        with open(filename, mode='wb') as f:
            n.tofile(f)


def save_img(img: np.ndarray, path: str, suffix: str) -> None:
    if len(path) == 0:
        path = "result.png"

    if len(suffix) == 0:
        print("Saving to", path)
        imwrite(path, img)
    else:
        file_path, ext = os.path.splitext(path)
        path = file_path + "_{}".format(suffix) + "." + ext
        print("Saving to", path)
        imwrite(path, img)


def read_images(pic_path: str, img_size: Tuple[int, int], recursive: bool = False,
                num_process: int = 1, flag: str = "stretch") -> List[np.ndarray]:
    assert os.path.isdir(pic_path), "Directory " + pic_path + "is non-existent"
    files = []
    if recursive:
        for root, subfolder, file_list in os.walk(pic_path):
            for f in file_list:
                files.append(os.path.join(root, f))
    else:
        root, _, file_names = list(os.walk(pic_path))[0]
        files = [os.path.join(root, f) for f in file_names]

    try:
        while True:
            files.remove("cache.pkl")
    except ValueError:
        pass

    pbar = tqdm(total=len(files),
                desc="[Reading files]", unit="file", ncols=pbar_ncols)

    if num_process <= 1:
        num_process = 1

    imgs = []
    slice_length = len(files) // num_process
    pool = con.ProcessPoolExecutor(num_process)
    futures = []
    queue = mp.Manager().Queue()
    for i in range(num_process - 1):
        futures.append(pool.submit(read_img_helper,
                                   files[i * slice_length:(i + 1) * slice_length], img_size, queue, flag))
    futures.append(pool.submit(read_img_helper,
                               files[(num_process - 1) * slice_length:], img_size, queue, flag))

    while True:
        try:
            out = queue.get_nowait()
            if isinstance(out, str):
                pbar.write(out)
            elif isinstance(out, int):
                pbar.update(out)
                if pbar.n >= len(files):
                    break
        except:
            time.sleep(0.001)

    for future in con.as_completed(futures):
        imgs.extend(future.result())

    pbar.close()

    return imgs


# this imread method can read images whose path contain unicode characters
def imread(filename: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_COLOR)


def read_img_helper(files: List[str], img_size: Tuple[int, int], queue: mp.Queue, flag: str = "center") -> List[np.ndarray]:
    imgs = []
    for img_file in files:
        try:
            img = imread(img_file)
            if flag == "center":
                h, w, _ = img.shape
                if w == h:
                    imgs.append(cv2.resize(img, img_size,
                                           interpolation=cv2.INTER_AREA))
                else:
                    if w > h:
                        ratio = img_size[1] / h
                        img = cv2.resize(
                            img, (round(w * ratio), img_size[1]), interpolation=cv2.INTER_AREA)
                        s = int((w * ratio - img_size[0]) // 2)
                        img = img[:, s:s + img_size[0], :]
                    else:
                        ratio = img_size[0] / w
                        img = cv2.resize(img, (img_size[0], round(
                            h * ratio)), interpolation=cv2.INTER_AREA)
                        s = int((h * ratio - img_size[1]) // 2)
                        img = img[s:s + img_size[1], :, :]
                    imgs.append(img)

            else:
                imgs.append(cv2.resize(img, img_size,
                                       interpolation=cv2.INTER_AREA))
        except:
            pass
        queue.put(1)
    return imgs


all_sort_methods = ["none", "bgr_sum", "av_hue", "av_sat", "av_lum", "rand"]

# these require scikit-learn
all_sort_methods.extend(["pca_bgr", "pca_hsv", "pca_lab", "pca_gray", "pca_lum", "pca_sat", "pca_hue",
                         "tsne_bgr", "tsne_hsv", "tsne_lab", "tsne_gray", "tsne_lum", "tsne_sat", "tsne_hue"])

# these require scikit-learn and umap-learn
# all_sort_methods.extend(["umap_bgr", "umap_hsv", "umap_lab", "umap_gray", "umap_lum", "umap_sat", "umap_hue"])

all_color_spaces = ["hsv", "hsl", "bgr", "lab"]

all_ctypes = ["float16", "float32", "float64"]

all_metrics = ["euclidean", "cityblock", "chebyshev"]

if __name__ == "__main__":
    mp.freeze_support()
    all_sigmas = np.concatenate(
        (np.arange(-1, -0.06, 0.05), np.arange(0.1, 1.05, 0.05)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the downloaded head images",
                        default=os.path.join(os.path.dirname(__file__), "img"), type=str)
    parser.add_argument("--recursive", action="store_true",
                        help="Whether to read the sub-folders of the designated folder")
    parser.add_argument("--num_process", type=int, default=1,
                        help="Number of processes to use when loading images")
    parser.add_argument("--out", default="", type=str,
                        help="The filename of the output image")
    parser.add_argument("--size", type=int, default=50,
                        help="Size of each image in pixels")
    parser.add_argument("--ratio", help="Aspect ratio",
                        nargs=2, type=int, default=(16, 9))
    parser.add_argument("--sort", help="Sort methods", choices=all_sort_methods,
                        type=str, default="bgr_sum")
    parser.add_argument("--rev_row",
                        help="Whether to use the S-shaped alignment. "
                             "Do NOT use this option when fitting an image using the --collage option",
                        action="store_true")
    parser.add_argument("--rev_sort",
                        help="Sort in the reverse direction. "
                             "Do NOT use this option when fitting an image using the --collage option",
                        action="store_true")
    parser.add_argument("--collage", type=str, default="",
                        help="If you want to fit an image, specify the image path here")
    parser.add_argument("--colorspace", type=str, default="lab", choices=all_color_spaces,
                        help="Methods to use when fitting an image")
    parser.add_argument("--metric", type=str, default="euclidean", choices=all_metrics,
                        help="Distance metric used when evaluating the distance between two color vectors")
    parser.add_argument("--uneven", action="store_true",
                        help="Whether to use each image only once")
    parser.add_argument("--max_width", type=int, default=80,
                        help="Maximum width of the collage")
    parser.add_argument("--dup", type=int, default=1,
                        help="Duplicate the set of images by how many times")
    parser.add_argument("--ctype", type=str, default="float16",
                        help="Type of the cost matrix. "
                             "Float16 is a good compromise between computational time and accuracy",
                        choices=all_ctypes)
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Add a weight to source images; a positive sigma implies a higher weight for the pixels in the middle of the image")
    parser.add_argument("--exp", action="store_true",
                        help="Traverse all possible options.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress message to console")
    parser.add_argument("--salient", action="store_true",
                        help="Collage salient object only")
    parser.add_argument("--lower_thresh", type=int, default=127)
    parser.add_argument("--background", nargs=3, type=int,
                        default=(225, 225, 225), help="Backgound color in RGB")

    args = parser.parse_args()
    if not args.verbose:
        sys.stdout = open(os.devnull, "w")

    if len(args.out) > 0:
        folder, file_name = os.path.split(args.out)
        if len(folder) > 0:
            assert os.path.isdir(
                folder), "The output path {} does not exist!".format(folder)
        # ext = os.path.splitext(file_name)[-1]
        # assert ext.lower() == ".jpg" or ext.lower() == ".png", "The file extension must be .jpg or .png"

    imgs = read_images(args.path, (args.size, args.size),
                       args.recursive, args.num_process)

    if len(args.collage) == 0:
        if args.exp:

            pool = con.ProcessPoolExecutor(4)
            futures = {}

            for sort_method in all_sort_methods:
                futures[pool.submit(sort_collage, imgs, args.ratio,
                                    sort_method, args.rev_sort)] = sort_method

            for future in tqdm(con.as_completed(futures.keys()), total=len(all_sort_methods), 
                               desc="[Experimenting]", unit="exps"):
                grid, sorted_imgs = future.result()
                combined_img = make_collage(grid, sorted_imgs, args.rev_row)
                save_img(combined_img, args.out, futures[future])

        else:
            grid, sorted_imgs = sort_collage(
                imgs, args.ratio, args.sort, args.rev_sort)
            save_img(make_collage(grid, sorted_imgs,
                                  args.rev_row), args.out, "")
    else:
        if args.exp:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt

            pool = con.ProcessPoolExecutor(5)
            futures = {}

            all_thresholds = list(range(40, 200, 10))

            if args.salient:
                if args.uneven:
                    for thresh in all_thresholds:
                        f = pool.submit(calc_salient_col_dup, args.collage,
                                        imgs, max_width=args.max_width, lower_thresh=thresh)
                        futures[f] = thresh
                else:
                    for thresh in all_thresholds:
                        f = pool.submit(
                            calc_salient_col_even_fast, args.collage, imgs, dup=args.dup, lower_thresh=thresh)
                        futures[f] = thresh
                
                cost_vis = {}
                for f in tqdm(con.as_completed(futures.keys()), desc="[Experimenting]", 
                              total=len(all_thresholds), unit="exps"):
                    thresh = futures[f]
                    suffix = "threshold_{}".format(f)
                    grid, sorted_imgs, cost = f.result()
                    save_img(make_collage(grid, sorted_imgs,
                                          args.rev_row), args.out, suffix)
                    cost_vis[thresh] = cost

                plt.figure()
                plt.plot(cost_vis.keys(), cost_vis.values())
                plt.xlabel("Threshold")
                plt.ylabel("Cost")
                plt.show()

            else:
                total_steps = len(all_sigmas) * len(all_color_spaces)

                if args.uneven:
                    for sigma in all_sigmas:
                        for color_space in all_color_spaces:
                            f = pool.submit(calc_col_dup, args.collage, imgs,
                                            args.max_width, color_space, sigma, args.metric)
                            futures[f] = (sigma, color_space)
                else:
                    for sigma in all_sigmas:
                        for color_space in all_color_spaces:
                            f = pool.submit(calc_col_even, args.collage, imgs, args.dup,
                                            color_space, args.ctype, sigma, args.metric)
                            futures[f] = (sigma, color_space)

                cost_vis = np.zeros((len(all_sigmas), len(all_color_spaces)))
                r, c = 0, 0
                for f in tqdm(con.as_completed(futures.keys()), desc="[Experimenting]", 
                              total=total_steps, unit="exps"):
                    sigma, color_space = futures[f]
                    suffix = "{}_{}".format(color_space, np.round(sigma, 2))
                    grid, sorted_imgs, cost = f.result()
                    save_img(make_collage(grid, sorted_imgs,
                                          args.rev_row), args.out, suffix)
                    cost_vis[r, c] = cost
                    c += 1
                    if c % len(all_color_spaces) == 0:
                        r += 1
                        c = 0

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                X, Y = np.meshgrid(
                    np.arange(len(all_color_spaces)), all_sigmas)
                surf = ax.plot_surface(X, Y, cost_vis, cmap="jet")
                ax.set_xlabel("Color Space")
                ax.set_ylabel("Sigma")
                ax.set_zlabel("Cost")
                plt.xticks(np.arange(len(all_color_spaces)), all_color_spaces)
                plt.show()

        else:
            if args.salient:
                if args.uneven:
                    grid, sorted_imgs, _ = calc_salient_col_dup(args.collage, imgs, args.max_width,
                                                                args.colorspace, args.sigma, args.metric,
                                                                args.lower_thresh, args.background)
                    save_img(make_collage(grid, sorted_imgs, args.rev_row),
                             args.out, "")
                else:
                    grid, sorted_imgs, _ = calc_salient_col_even_fast(args.collage, imgs, args.dup,
                                                                      args.colorspace, args.ctype, args.sigma,
                                                                      args.metric, args.lower_thresh, args.background)
                    save_img(make_collage(grid, sorted_imgs, args.rev_row),
                             args.out, "")
            else:
                if args.uneven:
                    grid, sorted_imgs, _ = calc_col_dup(args.collage, imgs, args.max_width,
                                                        args.colorspace, args.sigma, args.metric)
                    save_img(make_collage(grid, sorted_imgs, args.rev_row),
                             args.out, "")

                else:
                    grid, sorted_imgs, _ = calc_col_even(args.collage, imgs, args.dup,
                                                         args.colorspace, args.ctype, args.sigma, args.metric)
                    save_img(make_collage(grid, sorted_imgs, args.rev_row),
                             args.out, "")
