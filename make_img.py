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
import itertools
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


def calc_decay_weights_normal(shape: Tuple[int, int], sigma = 1.0, c = 3) -> np.ndarray:
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
        return np.ones((h, w, c))
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
            return np.repeat(weights[:, :, np.newaxis], c, axis=2)
        else:
            return np.ones((h, w, c))


def sort_collage(imgs: List[np.ndarray], ratio: Tuple[int, int], sort_method="pca_lab", rev_sort=False) -> Tuple[Tuple[int, int], np.ndarray]:
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
    print("Note that", num_imgs - grid[0] * grid[1], "images will be thrown away from the collage")
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


def chl_wise_avg_concat(imgs: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return np.array([np.average(img, axis=(0, 1), weights=weights) for img in imgs])
    

def calc_saliency_map(dest_img: np.ndarray, lower_thresh = 50,) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param dest_img: the destination image
    :param lower_thresh: lower threshold for salient object detection. 
                         If it's -1, then the threshold value will be adaptive
    :param fill_bg: whether to fill the background with the color specified
    :param bg_color: background color
    :return: [the copy of the destination image with background filled (if fill_bg), threshold map]
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

    return dest_img, thresh


def cvt_colorspace(colorspace: str, imgs: List[np.ndarray], dest_obj: np.ndarray):
    if colorspace == "hsv":
        flag = cv2.COLOR_BGR2HSV
    elif colorspace == "hsl":
        flag = cv2.cv2.COLOR_BGR2HLS
    elif colorspace == "bgr":
        return dest_obj, imgs
    elif colorspace == "lab":
        flag = cv2.COLOR_BGR2Lab
    else:
        raise ValueError("Unknown colorspace " + colorspace)
    return cv2.cvtColor(dest_obj, flag), [cv2.cvtColor(img, flag) for img in imgs]


def solve_lap(cost_matrix: np.ndarray, v=None):
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
    return cols, cost

def calc_salient_col_even_fast(dest_img_path: str, imgs: List[np.ndarray], dup=1, colorspace="lab", 
                               ctype ="float16", sigma = 1.0, metric = "euclidean", lower_thresh = 50,
                               background = (255, 255, 255), v=None) -> Tuple[Tuple[int, int], List[np.ndarray], float]:
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
    :param background: background color in rgb format
    :return: [gird size, sorted images, total assignment cost]
    """
    assert os.path.isfile(dest_img_path)

    t = time.time()
    print("Duplicating {} times".format(dup))

    # avoid modifying the original array
    imgs = list(map(np.copy, imgs))
    _copy = imgs.copy()
    for i in range(dup - 1):
        imgs.extend(_copy)

    dest_img = imread(dest_img_path)
    rh, rw, _ = dest_img.shape

    threshold = lower_thresh
    _, threshed_map = calc_saliency_map(dest_img, threshold)
    nonzero_idx = np.nonzero(threshed_map)

    dest_img_copy = np.copy(dest_img)
    pbar = tqdm(unit=" iteration", desc="[Computing saliency & grid]", ncols=pbar_ncols)
    while True:
        num_imgs = round(rh * rw / len(nonzero_idx[0]) * len(imgs))

        grid = calc_grid_size(rw, rh, num_imgs)

        dest_img = cv2.resize(dest_img_copy, grid, cv2.INTER_AREA)
        rh, rw, _ = dest_img.shape

        _, threshed_map = calc_saliency_map(dest_img, threshold)
        nonzero_idx = np.nonzero(threshed_map)
        diff = len(imgs) - len(nonzero_idx[0])

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
        if diff >= 0 and (diff < ceil(len(imgs) / dup / 3) or pbar.n > 50):
            break

    pbar.close()
    if threshold != -1:
        print("Final threshold:", threshold)

    print("Grid size {} calculated in {} iterations".format(grid, pbar.n))

    dest_obj = dest_img[nonzero_idx]
    dest_img[threshed_map == 0] = background[::-1]
    nonzero_idx = np.ravel_multi_index(nonzero_idx, threshed_map.shape) # flatten indices

    if len(imgs) > len(dest_obj):
        print("Note:", len(imgs) - len(dest_obj), "images will be thrown away from the collage")
    imgs = imgs[:len(dest_obj)]

    print("Computing cost matrix...")
    dest_obj, img_keys = cvt_colorspace(colorspace, imgs, dest_obj[np.newaxis])
    dest_obj = dest_obj[0]
    cost_matrix = cdist(chl_wise_avg_concat(img_keys, calc_decay_weights_normal(imgs[0].shape[:2], sigma)), dest_obj, metric=metric).astype(ctype)

    print("Computing optimal assignment on a {}x{} matrix...".format(cost_matrix.shape[0], cost_matrix.shape[1]))
    cols, cost = solve_lap(cost_matrix, v)
    del cost_matrix

    paired = np.array(imgs)[cols]

    white = np.ones(imgs[0].shape, np.uint8)
    white[:, :, :] = background[::-1]

    filled = []
    counter = 0
    for i in range(grid[0] * grid[1]):
        if i in nonzero_idx:
            filled.append(paired[counter])
            counter += 1
        else:
            filled.append(white)

    print("Total assignment cost:", cost)
    print("Time taken: {}s".format((np.round(time.time() - t, 2))))
    return grid, filled, cost


def calc_col_even(dest_img_path: str, imgs: List[np.ndarray], dup=1, colorspace="lab", 
                  ctype="float16", sigma=1.0, metric="euclidean", is_block=False, v=None) -> Tuple[Tuple[int, int], List[np.ndarray], float]:
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
    _copy = imgs.copy()
    for i in range(dup - 1):
        imgs.extend(_copy)

    num_imgs = len(imgs)

    # Compute the grid size based on the number images that we have
    dest_img = imread(dest_img_path)
    rh, rw, _ = dest_img.shape
    grid = calc_grid_size(rw, rh, num_imgs)
    total = grid[0] * grid[1]

    print("Calculated grid size based on the aspect ratio of the image provided:", grid)
    print("Note:", num_imgs - total, "images will be thrown away from the collage")

    # it's VERY important to remove redundant images
    # this makes sure that cost_matrix is a square
    imgs = imgs[:total]

    # Resize the destination image so that it has the same size as the grid
    # This makes sure that each image in the list of images corresponds to a pixel of the destination image
    if not is_block:
        dest_img = cv2.resize(dest_img, grid, interpolation=cv2.INTER_AREA)
    else:
        block_size = min(dest_img.shape[0] // grid[1], dest_img.shape[1] // grid[0])
        print("Block size:", block_size)
        dest_img = cv2.resize(dest_img, (grid[0] * block_size, grid[1] * block_size), interpolation=cv2.INTER_AREA)

    t = time.time()
    print("Computing cost matrix...")
    if not is_block:
        dest_img, img_keys = cvt_colorspace(colorspace, imgs, dest_img)
        dest_img = dest_img.reshape((total, 3))
        # compute pair-wise distances
        cost_matrix = cdist(chl_wise_avg_concat(img_keys, calc_decay_weights_normal(imgs[0].shape[:2], sigma)), dest_img, metric=metric).astype(ctype)
    else:
        dest_img, img_keys = cvt_colorspace(colorspace, [cv2.resize(img, (block_size, block_size), interpolation=cv2.INTER_AREA) for img in imgs], dest_img)
        n_dst_img = np.zeros((total, np.prod(img_keys[0].shape)), dtype=dest_img.dtype)
        k = 0
        for i in range(grid[1]):
            for j in range(grid[0]):
                n_dst_img[k, :] = dest_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, :].flatten()
                k += 1
        # compute pair-wise distances
        cost_matrix = cdist(np.array(img_keys).reshape((n_dst_img.shape)), n_dst_img, metric=metric).astype(ctype)

    print("Computing optimal assignment on a {}x{} matrix...".format(cost_matrix.shape[0], cost_matrix.shape[1]))
    cols, cost = solve_lap(cost_matrix, v)
    del cost_matrix # free memory

    print("Total assignment cost:", cost)
    print("Time taken: {}s".format((np.round(time.time() - t, 2))))
    return grid, np.array(imgs)[cols], cost


def solve_dup(dest_img: np.ndarray, img_keys: List[np.ndarray], imgs: List[np.ndarray], metric: str) -> Tuple[float, List[np.ndarray]]:
    cost = 0
    sorted_imgs = []
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
    return cost, sorted_imgs


def calc_salient_col_dup(dest_img_path: str, imgs: List[np.ndarray], max_width=80,
                         colorspace="lab", sigma=1.0, metric="euclidean", lower_thresh=127,
                         background=(255, 255, 255)) -> Tuple[Tuple[int, int], List[np.ndarray], float]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels that constitute 
    of the salient objects of the target image, given that every image could be used arbitrary amount of times

    :param dest_img_path: path to the dest_img file
    :param imgs: list of images
    :param max_width: max_width of the resulting dest_img
    :param colorspace: color space used
    :param sigma:
    :param lower_thresh: threshold for object detection
    :background background color in rgb format
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

    dest_img = cv2.resize(dest_img, grid, cv2.INTER_AREA)
    dest_img, thresh_map = calc_saliency_map(dest_img, lower_thresh)
    dest_img[thresh_map == 0] = background[::-1]

    white = np.ones(imgs[0].shape, np.uint8)
    white[:, :, :] = background[::-1]
    imgs.append(white)

    print("Computing costs...")
    dest_img, img_keys= cvt_colorspace(colorspace, imgs, dest_img)
    dest_img = dest_img.reshape(grid[0] * grid[1], 3)
    cost, sorted_imgs = solve_dup(dest_img, chl_wise_avg_concat(img_keys, calc_decay_weights_normal(imgs[0].shape[:2], sigma)), imgs, metric)

    print("Time taken: {}s".format(np.round(time.time() - t, 2)))
    return grid, sorted_imgs, cost


def calc_col_dup(dest_img_path: str, imgs: list, max_width=80, colorspace="lab",
                 sigma=1.0, metric="euclidean") -> Tuple[Tuple[int, int], List[np.ndarray], float]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels of the target image,
    given that every image could be used arbitrary amount of times

    :param dest_img_path: path to the dest_img file
    :param imgs: list of images
    :param max_width: max_width of the resulting dest_img
    :param colorspace: color space used
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

    t = time.time()
    print("Computing costs")
    dest_img, img_keys = cvt_colorspace(colorspace, imgs, cv2.resize(dest_img, grid, cv2.INTER_AREA))
    dest_img = dest_img.reshape(grid[0] * grid[1], 3)
    cost, sorted_imgs = solve_dup(dest_img, chl_wise_avg_concat(img_keys, calc_decay_weights_normal(imgs[0].shape[:2], sigma)), imgs, metric)

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


def read_images(pic_path: str, img_size: Tuple[int, int], recursive=False, num_process=1, flag="stretch") -> List[np.ndarray]:
    assert os.path.isdir(pic_path), "Directory " + pic_path + "is non-existent"
    files = []
    for root, _, file_list in os.walk(pic_path):
        for f in file_list:
            files.append(os.path.join(root, f))
        if not recursive:
            break

    if num_process <= 1:
        num_process = 1
    pool = mp.Pool(num_process)
    func = read_img_center if flag == "center" else read_img_other
    return [
        r for r in tqdm(
            pool.imap_unordered(func, zip(files, itertools.repeat(img_size, len(files))), chunksize=64), 
            total=len(files), desc="[Reading files]", unit="file", ncols=pbar_ncols) if r is not None
    ]


# this imread method can read images whose path contain unicode characters
def imread(filename: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_COLOR)


def read_img_center(args: Tuple[str, Tuple[int, int]]):
    img_file, img_size = args
    try:
        img = imread(img_file)
        h, w, _ = img.shape
        if w == h:
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        elif w > h:
            ratio = img_size[1] / h
            img = cv2.resize(img, (round(w * ratio), img_size[1]), interpolation=cv2.INTER_AREA)
            s = int((w * ratio - img_size[0]) / 2)
            img = img[:, s:s + img_size[0], :]
        else:
            ratio = img_size[0] / w
            img = cv2.resize(img, (img_size[0], round(h * ratio)), interpolation=cv2.INTER_AREA)
            s = int((h * ratio - img_size[1]) / 2)
            img = img[s:s + img_size[1], :, :]
        return img
    except:
        return None


def read_img_other(args: Tuple[str, Tuple[int, int]]):
    img_file, img_size = args
    try:
        img = imread(img_file)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        return img
    except:
        return None


all_sort_methods = ["none", "bgr_sum", "av_hue", "av_sat", "av_lum", "rand"]

# these require scikit-learn
all_sort_methods.extend(["pca_bgr", "pca_hsv", "pca_lab", "pca_gray", "pca_lum", "pca_sat", "pca_hue",
                         "tsne_bgr", "tsne_hsv", "tsne_lab", "tsne_gray", "tsne_lum", "tsne_sat", "tsne_hue"])

# these require scikit-learn and umap-learn
# all_sort_methods.extend(["umap_bgr", "umap_hsv", "umap_lab", "umap_gray", "umap_lum", "umap_sat", "umap_hue"])

all_colorspaces = ["hsv", "hsl", "bgr", "lab"]

all_ctypes = ["float16", "float32", "float64"]

all_metrics = ["euclidean", "cityblock", "chebyshev"]

if __name__ == "__main__":
    mp.freeze_support()
    all_sigmas = np.concatenate((np.arange(-1, -0.06, 0.05), np.arange(0.1, 1.05, 0.05)))

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
    parser.add_argument("--colorspace", type=str, default="lab", choices=all_colorspaces,
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
                        default=(255, 255, 255), help="Background color in RGB")

    args = parser.parse_args()
    if not args.verbose:
        sys.stdout = open(os.devnull, "w")

    if len(args.out) > 0:
        folder, file_name = os.path.split(args.out)
        if len(folder) > 0:
            assert os.path.isdir(folder), "The output path {} does not exist!".format(folder)
        # ext = os.path.splitext(file_name)[-1]
        # assert ext.lower() == ".jpg" or ext.lower() == ".png", "The file extension must be .jpg or .png"

    imgs = read_images(args.path, (args.size, args.size), args.recursive, args.num_process)

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
                total_steps = len(all_sigmas) * len(all_colorspaces)

                if args.uneven:
                    for sigma in all_sigmas:
                        for colorspace in all_colorspaces:
                            f = pool.submit(calc_col_dup, args.collage, imgs,
                                            args.max_width, colorspace, sigma, args.metric)
                            futures[f] = (sigma, colorspace)
                else:
                    for sigma in all_sigmas:
                        for colorspace in all_colorspaces:
                            f = pool.submit(calc_col_even, args.collage, imgs, args.dup,
                                            colorspace, args.ctype, sigma, args.metric)
                            futures[f] = (sigma, colorspace)

                cost_vis = np.zeros((len(all_sigmas), len(all_colorspaces)))
                r, c = 0, 0
                for f in tqdm(con.as_completed(futures.keys()), desc="[Experimenting]",
                              total=total_steps, unit="exps"):
                    sigma, colorspace = futures[f]
                    suffix = "{}_{}".format(colorspace, np.round(sigma, 2))
                    grid, sorted_imgs, cost = f.result()
                    save_img(make_collage(grid, sorted_imgs, args.rev_row), args.out, suffix)
                    cost_vis[r, c] = cost
                    c += 1
                    if c % len(all_colorspaces) == 0:
                        r += 1
                        c = 0

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                X, Y = np.meshgrid(np.arange(len(all_colorspaces)), all_sigmas)
                surf = ax.plot_surface(X, Y, cost_vis, cmap="jet")
                ax.set_xlabel("Color Space")
                ax.set_ylabel("Sigma")
                ax.set_zlabel("Cost")
                plt.xticks(np.arange(len(all_colorspaces)), all_colorspaces)
                plt.show()

        else:
            if args.salient:
                if args.uneven:
                    grid, sorted_imgs, _ = calc_salient_col_dup(args.collage, imgs, args.max_width,
                                                                args.colorspace, args.sigma, args.metric,
                                                                args.lower_thresh, args.background)
                    save_img(make_collage(grid, sorted_imgs, args.rev_row), args.out, "")
                else:
                    grid, sorted_imgs, _ = calc_salient_col_even_fast(args.collage, imgs, args.dup,
                                                                      args.colorspace, args.ctype, args.sigma,
                                                                      args.metric, args.lower_thresh, args.background)
                    save_img(make_collage(grid, sorted_imgs, args.rev_row), args.out, "")
            else:
                if args.uneven:
                    grid, sorted_imgs, _ = calc_col_dup(args.collage, imgs, args.max_width,
                                                        args.colorspace, args.sigma, args.metric)
                    save_img(make_collage(grid, sorted_imgs, args.rev_row), args.out, "")

                else:
                    grid, sorted_imgs, _ = calc_col_even(args.collage, imgs, args.dup,
                                                         args.colorspace, args.ctype, args.sigma, args.metric)
                    save_img(make_collage(grid, sorted_imgs, args.rev_row), args.out, "")
