import os
from os.path import *
import numpy as np
import cv2
import argparse
import random
from tqdm import tqdm
import concurrent.futures as con
import multiprocessing as mp
import sys
import time
import platform
from typing import *
import traceback
from math import ceil

pbar_ncols = None


def bgr_chl_sum(img: np.ndarray) -> Tuple[float, float, float]:
    return np.sum(img[:, :, 0]), np.sum(img[:, :, 1]), np.sum(img[:, :, 2])


def bgr_sum(img: np.ndarray) -> float:
    return np.sum(img)


def av_hue(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0])


def av_sat(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 1])


def hsv(img: np.ndarray) -> Tuple[float, float, float]:
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
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab).flatten()


def pca_hue(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0].flatten()


def pca_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()


def rand(img: np.ndarray) -> float:
    return random.random()


# The output wrapper for displaying the progress of J-V algorithm is only available for the GUI running in Linux
class JVOutWrapper:
    def __init__(self, io_wrapper):
        self.io_wrapper = io_wrapper
        self.tqdm = None

    def write(self, lines):
        lines = lines.split("\n")
        for line in lines:
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


def calculate_grid_size(rw: int, rh: int, num_imgs: int) -> Tuple[int, int]:
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


def calculate_decay_weights_normal(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """
    Generate a matrix of probabilities (as weights) sampled from a truncated bivariate normal distribution
    centered at (0, 0) whose covariance matrix is equal to sigma times the identity matrix

    Basically it's a product of two identical and independent normal distributions truncated in the range [-1,1]
    with mean=0 and sigma=sigma

    Generate an "upside-down" bivariate normal if sigma is negative.
    """
    assert sigma != 0
    from scipy.stats import truncnorm
    h, w = shape
    h_arr = truncnorm.pdf(np.linspace(-1, 1, h), -1,
                          1, loc=0, scale=abs(sigma))
    w_arr = truncnorm.pdf(np.linspace(-1, 1, w), -1,
                          1, loc=0, scale=abs(sigma))
    h_arr, w_arr = np.meshgrid(h_arr, w_arr)
    if sigma > 0:
        weights = h_arr * w_arr
    else:
        weights = 1 - h_arr * w_arr
    return weights


def sort_collage(imgs: List[np.ndarray], ratio: Tuple[int, int], sort_method="pca_lab", rev_sort=False) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    :param imgs: list of images
    :param ratio: The aspect ratio of the collage
    :param sort_method:
    :param rev_sort: whether to reverse the sorted array
    :return: calculated grid size and the sorted image array
    """
    num_imgs = len(imgs)
    grid = calculate_grid_size(ratio[0], ratio[1], num_imgs)

    print("Calculated grid size based on your aspect ratio:", grid)
    print("Note that", num_imgs - grid[0] * grid[1],
          "images will be thrown away from the collage")
    print("Sorting images...")
    t = time.clock()
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
        if type(img_keys[0]) == tuple:
            img_keys = list(map(lambda x: x[0], img_keys))
        img_keys = np.array(img_keys)

    sorted_imgs = np.array(imgs)[np.argsort(img_keys)]
    if rev_sort:
        sorted_imgs = list(reversed(sorted_imgs))

    print("Time taken: {}s".format(np.round(time.clock() - t, 2)))

    return grid, sorted_imgs


def chl_mean_hsv(weights: np.ndarray) -> Callable:
    def f(img: np.ndarray) -> Tuple[float, float, float]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.average(img[:, :, 0], weights=weights), \
            np.average(img[:, :, 1], weights=weights), \
            np.average(img[:, :, 2], weights=weights)

    return f


def chl_mean_hsl(weights: np.ndarray) -> Callable:
    def f(img: np.ndarray) -> Tuple[float, float, float]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return np.average(img[:, :, 0], weights=weights), \
            np.average(img[:, :, 1], weights=weights), \
            np.average(img[:, :, 2], weights=weights)

    return f


def chl_mean_bgr(weights: np.ndarray) -> Callable:
    def f(img: np.ndarray) -> Tuple[float, float, float]:
        return np.average(img[:, :, 0], weights=weights), \
            np.average(img[:, :, 1], weights=weights), \
            np.average(img[:, :, 2], weights=weights)

    return f


def chl_mean_lab(weights: np.ndarray) -> Callable:
    def f(img: np.ndarray) -> Tuple[float, float, float]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return np.average(img[:, :, 0], weights=weights), \
            np.average(img[:, :, 1], weights=weights), \
            np.average(img[:, :, 2], weights=weights)

    return f


def calculate_salient_collage_bipartite(dest_img_path: str, imgs: List[np.ndarray], dup: int = 1,
                                        colorspace: str = "lab", ctype: str = "float16", sigma: float = 1.0,
                                        metric: str = "euclidean", v=None) -> Tuple[Tuple[int, int], List[np.ndarray], float]:
    assert isfile(dest_img_path)
    from scipy.spatial.distance import cdist

    print("Duplicating {} times".format(dup))

    # avoid modifying the original array
    imgs = list(map(np.copy, imgs))
    imgs_copy = list(map(np.copy, imgs))
    for i in range(dup - 1):
        imgs.extend(imgs_copy)

    num_imgs = len(imgs)

    # Compute the grid size based on the number images that we have
    dest_img = cv2.imread(dest_img_path)

    # Remove un-salient part of dest_image
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (_, saliency_map) = saliency.computeSaliency(dest_img)
    (_, thresh) = cv2.threshold(saliency_map * 255, 50, 255, cv2.THRESH_BINARY)

    # print(thresh)

    rh, rw, _ = dest_img.shape

    total_area = rh * rw
    obj_area = 0

    for i in range(rh):
        for j in range(rw):
            if int(thresh[i][j]) != 0:
                obj_area += 1
            else:
                dest_img[i][j] = [255, 255, 255]
    
    print("total area is {}; object area is {}".format(total_area, obj_area))

    num_white = ceil(num_imgs * dup / obj_area * (total_area - obj_area))

    h, w, _ = imgs[0].shape

    imgs.extend([np.ones((h, w, 3), np.uint8) * 255 for i in range(num_white)])

    num_imgs = len(imgs)

    grid = calculate_grid_size(rw, rh, num_imgs)

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





    weights = calculate_decay_weights_normal(imgs[0].shape[:2], sigma)
    t = time.clock()
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
    print("Time taken: {}s".format(np.round(time.clock() - t), 2))

    # sometimes the cost matrix may be extremely large
    # manually delete it to free memory
    del cost_matrix

    return grid, np.array(imgs)[cols], cost


def calculate_collage_bipartite(dest_img_path: str, imgs: List[np.ndarray], dup: int = 1,
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
    assert isfile(dest_img_path)
    from scipy.spatial.distance import cdist

    print("Duplicating {} times".format(dup))

    # avoid modifying the original array
    imgs = list(map(np.copy, imgs))
    imgs_copy = list(map(np.copy, imgs))
    for i in range(dup - 1):
        imgs.extend(imgs_copy)

    num_imgs = len(imgs)

    # Compute the grid size based on the number images that we have
    dest_img = cv2.imread(dest_img_path)
    rh, rw, _ = dest_img.shape
    grid = calculate_grid_size(rw, rh, num_imgs)

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

    weights = calculate_decay_weights_normal(imgs[0].shape[:2], sigma)
    t = time.clock()
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
    print("Time taken: {}s".format((np.round(time.clock() - t), 2)))

    # sometimes the cost matrix may be extremely large
    # manually delete it to free memory
    del cost_matrix

    return grid, np.array(imgs)[cols], cost

def calculate_salient_collage_dup(dest_img_path: str, imgs: list, max_width: int = 50, color_space="lab",
                          sigma: float = 1.0, metric: str = "euclidean") -> Tuple[Tuple[int, int], List[np.ndarray], float]:
    assert isfile(dest_img_path)
    from scipy.spatial.distance import cdist

    dest_img = cv2.imread(dest_img_path)

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency.computeSaliency(dest_img)
    _, thresh = cv2.threshold(saliency_map * 255, 50, 255, cv2.THRESH_BINARY)

    rh, rw, _ = dest_img.shape

    # print(thresh)

    for i in range(rh):
        for j in range(rw):
            if thresh[i][j] < 10:
                # print("haha ")
                dest_img[i][j] = [255,255,255]

    print(dest_img)

    rh = round(rh * max_width / rw)
    grid = (max_width, rh)

    h, w, _ = imgs[0].shape

    # print(np.ones((h, w, 3), np.uint8) * 255)
    imgs.append(np.ones((h, w, 3), np.uint8) * 255)

    print("Calculated grid size based on the aspect ratio of the image provided:", grid)

    weights = calculate_decay_weights_normal(imgs[0].shape[:2], sigma)
    dest_img = cv2.resize(dest_img, grid, cv2.INTER_AREA)
    t = time.clock()
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

    print("Time taken: {}s".format(np.round(time.clock() - t, 2)))

    return grid, sorted_imgs, cost
    

def calculate_collage_dup(dest_img_path: str, imgs: list, max_width: int = 50, color_space="lab",
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
    assert isfile(dest_img_path)
    from scipy.spatial.distance import cdist

    dest_img = cv2.imread(dest_img_path)

    # Because we don't have a fixed total amount of images as we can used a single image
    # for arbitrary amount of times, we need user to specify the maximum width in order to determine the grid size.
    rh, rw, _ = dest_img.shape
    rh = round(rh * max_width / rw)
    grid = (max_width, rh)

    print("Calculated grid size based on the aspect ratio of the image provided:", grid)

    weights = calculate_decay_weights_normal(imgs[0].shape[:2], sigma)
    dest_img = cv2.resize(dest_img, grid, cv2.INTER_AREA)
    t = time.clock()
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

    print("Time taken: {}s".format(np.round(time.clock() - t, 2)))

    return grid, sorted_imgs, cost


def save_img(img: np.ndarray, path: str, suffix: str) -> None:
    def imwrite(filename, img):
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img)

        if result:
            with open(filename, mode='wb') as f:
                n.tofile(f)

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
            if type(out) == str:
                pbar.write(out)
            elif type(out) == int:
                pbar.update(out)
                if pbar.n >= len(files):
                    break
        except:
            time.sleep(0.001)

    for future in con.as_completed(futures):
        imgs.extend(future.result())

    pbar.close()

    return imgs


def read_img_helper(files: List[str], img_size: Tuple[int, int], queue: mp.Queue, flag: str = "center") -> List[np.ndarray]:
    def imread(filename):
        return cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_COLOR)

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
        except Exception as e:
            pass
            # queue.put(traceback.format_exc())
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
        (np.arange(-1, -0.45, 0.05), np.arange(0.5, 1.05, 0.05)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the downloaded head images",
                        default=join(dirname(__file__), "img"), type=str)
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
    parser.add_argument("--max_width", type=int, default=50,
                        help="Maximum width of the collage")
    parser.add_argument("--dup", type=int, default=1,
                        help="Duplicate the set of images by how many times")
    parser.add_argument("--ctype", type=str, default="float16",
                        help="Type of the cost matrix. "
                             "Float16 is a good compromise between computational time and accuracy",
                        choices=all_ctypes)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--exp", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--salient", action="store_true")

    args = parser.parse_args()
    if not args.verbose:
        sys.stdout = open(devnull, 'w')
        sys.stderr = sys.stdout

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

            for future in tqdm(con.as_completed(futures.keys()), total=len(all_sort_methods), desc="[Experimenting]",
                               unit="exps", file=sys.__stderr__):
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
            # noinspection PyUnresolvedReferences
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt

            pool = con.ProcessPoolExecutor(4)
            futures = []

            total_steps = len(all_sigmas) * len(all_color_spaces)

            if args.uneven:
                for sigma in all_sigmas:
                    for color_space in all_color_spaces:
                        f = pool.submit(calculate_collage_dup, args.collage, imgs,
                                        args.max_width, color_space, sigma, args.metric)
                        futures.append((f, sigma, color_space))
            # elif args.salient:
            #     for sigma in all_sigmas:
            #         for color_space in all_color_spaces:
            #             f = pool.submit(calculate_salient_collage_bipartite, args.collage, imgs, args.dup,
            #                             color_space, args.ctype, sigma, args.metric)
            #             futures.append((f, sigma, color_space))
            #             #calculate_salient_collage_bipartite
            else:
                for sigma in all_sigmas:
                    for color_space in all_color_spaces:
                        f = pool.submit(calculate_collage_bipartite, args.collage, imgs, args.dup,
                                        color_space, args.ctype, sigma, args.metric)
                        futures.append((f, sigma, color_space))

            cost_vis = np.zeros((len(all_sigmas), len(all_color_spaces)))
            r, c = 0, 0
            for (f, sigma, color_space) in tqdm(futures, desc="[Experimenting]", unit="exps", file=sys.__stdout__):
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
            X, Y = np.meshgrid(np.arange(len(all_color_spaces)), all_sigmas)
            surf = ax.plot_surface(X, Y, cost_vis)
            ax.set_xlabel("Color Space")
            ax.set_ylabel("Sigma")
            ax.set_zlabel("Cost")
            plt.xticks(np.arange(len(all_color_spaces)), all_color_spaces)
            plt.show()

        else:
            if args.uneven and args.salient:
                grid, sorted_imgs, _ = calculate_salient_collage_dup(args.collage, imgs, args.max_width,
                                                             args.colorspace, args.sigma, args.metric)
                save_img(make_collage(grid, sorted_imgs, args.rev_row),
                         args.out, "")
            if args.uneven:
                grid, sorted_imgs, _ = calculate_collage_dup(args.collage, imgs, args.max_width,
                                                             args.colorspace, args.sigma, args.metric)
                save_img(make_collage(grid, sorted_imgs, args.rev_row),
                         args.out, "")
            elif args.salient:
                grid, sorted_imgs, _ = calculate_salient_collage_bipartite(args.collage, imgs, args.dup,
                                                                           args.colorspace, args.ctype, args.sigma, args.metric)
                save_img(make_collage(grid, sorted_imgs, args.rev_row),
                         args.out, "")
            else:
                grid, sorted_imgs, _ = calculate_collage_bipartite(args.collage, imgs, args.dup,
                                                                   args.colorspace, args.ctype, args.sigma, args.metric)
                save_img(make_collage(grid, sorted_imgs, args.rev_row),
                         args.out, "")
