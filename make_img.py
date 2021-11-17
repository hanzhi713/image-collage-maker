import os
import argparse
import random
import concurrent.futures as con
import multiprocessing as mp
import sys
import time
import platform
from typing import Any, List, Tuple
import itertools

import cv2
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if mp.current_process().name != "MainProcess":
    sys.stdout = open(os.devnull, "w")
    sys.stderr = sys.stdout

pbar_ncols = None


class _PARAMETER:
    def __init__(self, type: Any, help: str, default=None, nargs=None, choices: List[Any]=None) -> None:
        self.type = type
        self.default = default
        self.help = help
        self.nargs = nargs
        self.choices = choices

# We gather parameters here so they can be reused else where
class PARAMS:
    path = _PARAMETER(help="Path to the tiles", default=os.path.join(os.path.dirname(__file__), "img"), type=str)
    recursive = _PARAMETER(type=bool, default=False, help="Whether to read the sub-folders for the specified path")
    num_process = _PARAMETER(type=int, default=mp.cpu_count() // 2, help="Number of processes to use when loading images")
    out = _PARAMETER(default="", type=str, help="The filename of the output image")
    size = _PARAMETER(type=int, default=50, help="Size (side length) of each tile in pixels in the resulting collage/photomosaic")
    verbose = _PARAMETER(type=bool, default=False, help="Print progress message to console")
    resize_opt = _PARAMETER(type=str, default="center", choices=["center", "stretch"], 
        help="How to resize each tile so they become square images. "
             "Center: crop a square in the center. Stretch: stretch the tile")

    # ---------------- sort collage options ------------------
    ratio = _PARAMETER(type=int, default=(16, 9), help="Aspect ratio of the output image", nargs=2)
    sort = _PARAMETER(type=str, default="bgr_sum", help="Sort method to use", choices=[
        "none", "bgr_sum", "av_hue", "av_sat", "av_lum", "rand",
        "pca_bgr", "pca_hsv", "pca_lab", "pca_gray", "pca_lum", "pca_sat", "pca_hue",
        "tsne_bgr", "tsne_hsv", "tsne_lab", "tsne_gray", "tsne_lum", "tsne_sat", "tsne_hue"
    ])
    rev_row = _PARAMETER(type=bool, default=False, help="Whether to use the S-shaped alignment.")
    rev_sort = _PARAMETER(type=bool, default=False, help="Sort in the reverse direction.")
    
    # ---------------- photomosaic common options ------------------
    dest_img = _PARAMETER(type=str, default="", help="The path to the destination image that you want to build a photomosaic for")
    colorspace = _PARAMETER(type=str, default="lab", choices=["hsv", "hsl", "bgr", "lab", "luv"], 
        help="The colorspace used to calculate the metric")
    metric = _PARAMETER(type=str, default="euclidean", choices=["euclidean", "cityblock", "chebyshev"], 
        help="Distance metric used when evaluating the distance between two color vectors")
    ctype = _PARAMETER(type=str, default="float32", choices=["float32", "float64"],
        help="C type of the cost matrix. float32 is a good compromise between computational time and accuracy. Leave as default if unsure.")
    
    # ---- unfair tile assginment options -----
    unfair = _PARAMETER(type=bool, default=False, 
        help="Whether to allow each tile to be used different amount of times (unfair tile usage). ")
    max_width = _PARAMETER(type=int, default=80, help="Maximum width of the collage. This option is only valid if unfair option is enabled")    
    freq_mul = _PARAMETER(type=int, default=1, 
        help="Frequency multiplier to balance tile fairless and mosaic quality. Minimum: 0. "
             "More weight will be put on tile fairness when this number increases.")
    deterministic = _PARAMETER(type=bool, default=False, help="Do not randomize the tiles for unfair tile usage")

    # --- fair tile assignment options ---
    dup = _PARAMETER(type=int, default=1, help="Duplicate the set of tiles by how many times")

    # ---- saliency detection options ---
    salient = _PARAMETER(type=bool, default=False, help="Make photomosaic for salient objects only")
    lower_thresh = _PARAMETER(type=float, default=0.5, help="The threshold for saliency detection, between 0.0 and 1.0")
    background = _PARAMETER(nargs=3, type=int, default=(255, 255, 255), 
        help="Background color in RGB for non salient part of the image")


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


def make_collage(grid: Tuple[int, int], sorted_imgs: List[np.ndarray], rev=False) -> np.ndarray:
    """
    :param grid: grid size
    :param sorted_imgs: list of images sorted in correct position
    :param rev: whether to have opposite alignment for consecutive rows
    :return: a collage
    """
    print("Aligning images on the grid...")
    combined_img = np.asarray(sorted_imgs[:np.prod(grid)])
    combined_img.shape = (*grid[::-1], *sorted_imgs[0].shape)
    if rev:
        combined_img[1::2] = combined_img[1::2, ::-1]
    combined_img = combined_img.transpose((0, 2, 1, 3, 4))
    combined_img = combined_img.reshape(np.prod(combined_img.shape[:2]), -1, 3)
    return combined_img


def alpha_blend(combined_img: np.ndarray, dest_img: np.ndarray, alpha=0.9):
    dest_img = cv2.resize(dest_img, combined_img.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    dest_img *= 1 - alpha
    combined_img = combined_img * alpha # copy
    combined_img += dest_img
    return combined_img


def lightness_blend(combined_img: np.ndarray, dest_img: np.ndarray, alpha=0.9):
    dest_img = cv2.resize(dest_img, combined_img.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    cv2.cvtColor(dest_img, cv2.COLOR_BGR2HLS, dst=dest_img)
    dest_img[:, :, 1] *= 1 - alpha
    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2HLS)
    combined_img[:, :, 1] *= alpha
    combined_img[:, :, 1] += dest_img[:, :, 1]
    cv2.cvtColor(combined_img, cv2.COLOR_HLS2BGR, dst=combined_img)
    return combined_img


def sort_collage(imgs: List[np.ndarray], ratio: Tuple[int, int], sort_method="pca_lab", rev_sort=False) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    :param imgs: list of images
    :param ratio: The aspect ratio of the collage
    :param sort_method:
    :param rev_sort: whether to reverse the sorted array
    :return: [calculated grid size, sorted image array]
    """
    t = time.time()
    num_imgs = len(imgs)
    grid = calc_grid_size(ratio[0], ratio[1], num_imgs)

    print("Calculated grid size based on your aspect ratio:", grid)
    print("Note that", num_imgs - grid[0] * grid[1], "images will be thrown away from the collage")
    print("Sorting images...")

    if sort_method == "none":
        return grid, imgs
    
    if sort_method.startswith("pca_"):
        sort_function = eval(sort_method)
        img_keys = PCA(1).fit_transform(list(map(sort_function, imgs)))[:, 0]
    elif sort_method.startswith("tsne_"):
        sort_function = eval(sort_method.replace("tsne", "pca"))
        img_keys = TSNE(n_components=1, verbose=1, init="pca").fit_transform(
            list(map(sort_function, imgs)))[:, 0]
    else:
        sort_function = eval(sort_method)
        img_keys = np.array(list(map(sort_function, imgs)))

    indices = np.argsort(img_keys)
    if rev_sort:
        indices = indices[::-1]
    print("Time taken: {}s".format(np.round(time.time() - t, 2)))
    return grid, [imgs[i] for i in indices]


def cvt_colorspace(colorspace: str, imgs: List[np.ndarray], dest_img: np.ndarray):
    normalize_first = False
    if colorspace == "bgr":
        return
    elif colorspace == "hsv":
        flag = cv2.COLOR_BGR2HSV
        normalize_first = True
    elif colorspace == "hsl":
        flag = cv2.COLOR_BGR2HLS
        normalize_first = True
    elif colorspace == "lab":
        flag = cv2.COLOR_BGR2LAB
    elif colorspace == "luv":
        flag = cv2.COLOR_BGR2LUV
    else:
        raise ValueError("Unknown colorspace " + colorspace)
    for img in imgs:
        cv2.cvtColor(img, flag, dst=img)
    cv2.cvtColor(dest_img, flag, dst=dest_img)
    if normalize_first:
        # for hsv/hsl, h is in range 0~360 while other channels are in range 0~1
        # need to normalize
        for img in imgs:
            img[:, :, 0] *= 1 / 360.0
        dest_img[:, :, 0] *= 1 / 360.0


def solve_lap(cost_matrix: np.ndarray, v=None):
    print("Computing optimal assignment on a {}x{} matrix...".format(cost_matrix.shape[0], cost_matrix.shape[1]))
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
    return cols, cost


def compute_block_map(thresh_map, block_size, lower_thresh):
    height, width = thresh_map.shape
    dst_size = (width - width % block_size, height - height % block_size)
    if thresh_map.shape[::-1] != dst_size:
        thresh_map = cv2.resize(thresh_map, dst_size, interpolation=cv2.INTER_AREA)
    row_idx, col_idx = np.nonzero(thresh_map.reshape(
        dst_size[1] // block_size, block_size, dst_size[0] // block_size, block_size).max(axis=(1, 3)) >= lower_thresh
    )
    return row_idx, col_idx, thresh_map


def compute_blocks_salient(colorspace: str, dest_img: np.ndarray, imgs: List[np.ndarray], block_size: int, ridx, cidx, thresh_map, lower_thresh, background):
    dest_img = cv2.resize(dest_img, thresh_map.shape[::-1], interpolation=cv2.INTER_AREA)
    bg = np.asarray(background[::-1], dtype=imgs[0].dtype)
    bg *= 1 / 255.0
    dest_img[thresh_map < lower_thresh] = bg

    img_keys = [cv2.resize(img, (block_size, block_size), interpolation=cv2.INTER_AREA) for img in imgs]
    cvt_colorspace(colorspace, img_keys, dest_img)
    dest_img.shape = (thresh_map.shape[0] // block_size, block_size, thresh_map.shape[1] // block_size, block_size, 3)
    dest_img = dest_img[ridx, :, cidx, :, :].reshape(-1, block_size * block_size * 3)
    return dest_img, np.array(img_keys).reshape(-1, dest_img.shape[1]), np.full(imgs[0].shape, bg, dtype=imgs[0].dtype)


def calc_salient_col_even(dest_img: np.ndarray, imgs: List[np.ndarray], dup=1, colorspace="lab", 
                               ctype="float32", metric="euclidean", lower_thresh=0.5,
                               background=(255, 255, 255), v=None) -> Tuple[Tuple[int, int], List[np.ndarray]]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels constitute of salient objects of the
    target image, with the restriction that every image should be used the same amount of times

    non salient part of the target image is filled with background color=background
    """
    t = time.time()
    print("Duplicating {} times".format(dup))
    imgs = imgs * dup
    height, width, _ = dest_img.shape

    # this is just the initial grid size
    grid = calc_grid_size(width, height, len(imgs))
    block_size = min(width // grid[0], height // grid[1])
    prev_block_size = block_size
    _, thresh_map = cv2.saliency.StaticSaliencyFineGrained_create().computeSaliency((dest_img * 255).astype(np.uint8))

    while True:
        ridx, _, _ = compute_block_map(thresh_map, block_size, lower_thresh)
        if len(ridx) >= len(imgs):
            break
        prev_block_size = block_size
        block_size -= 1
    
    block_size = prev_block_size
    ridx, cidx, thresh_map = compute_block_map(thresh_map, block_size, lower_thresh)
    print("Block size:", block_size)

    grid = (width // block_size, height // block_size)
    print("Grid size:", grid)
    assert len(ridx) <= len(imgs)
    del imgs[len(ridx):]

    dest_img, img_keys, bg_img = compute_blocks_salient(colorspace, dest_img, imgs, block_size, ridx, cidx, thresh_map, lower_thresh, background)

    print("Computing cost matrix...")
    cols, cost = solve_lap(cdist(img_keys, dest_img, metric=metric).astype(ctype), v)

    imgs.append(bg_img)
    assignment = np.full(grid[::-1], len(imgs) - 1, dtype=np.int32)
    assignment[ridx, cidx] = cols
    assignment = assignment.flatten()
    print("Time taken: {}s".format((np.round(time.time() - t, 2))))
    return grid, [imgs[i] for i in assignment]


def compute_blocks(colorspace: str, dest_img: np.ndarray, imgs: List[np.ndarray], grid: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    block_size = min(dest_img.shape[0] // grid[1], dest_img.shape[1] // grid[0])
    print("Block size:", block_size)
    dest_img = cv2.resize(dest_img, (grid[0] * block_size, grid[1] * block_size), interpolation=cv2.INTER_AREA)
    img_keys = [cv2.resize(img, (block_size, block_size), interpolation=cv2.INTER_AREA) for img in imgs]
    cvt_colorspace(colorspace, img_keys, dest_img)
    flat_block_size = block_size * block_size * 3
    return dest_img.reshape(grid[1], block_size, grid[0], block_size, 3).transpose((0, 2, 1, 3, 4)).reshape(-1, flat_block_size), np.array(img_keys).reshape(-1, flat_block_size)


def calc_col_even(dest_img: np.ndarray, imgs: List[np.ndarray], dup=1, colorspace="lab", 
                  ctype="float32", metric="euclidean", v=None, grid=None) -> Tuple[Tuple[int, int], List[np.ndarray]]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels of the target image,
    with the restriction that every image should be used the same amount of times
    """

    t = time.time()

    if grid is not None:
        print("Use the provided grid size:", grid)
        dup = np.prod(grid) // len(imgs) + 1
    else:
        # Compute the grid size based on the number images that we have
        grid = calc_grid_size(dest_img.shape[1],  dest_img.shape[0], len(imgs) * dup)
        print("Calculated grid size based on the aspect ratio of the image provided:", grid)
    total = grid[0] * grid[1]

    print("Duplicated {} times".format(dup))
    imgs = imgs * dup

    print("Note:", len(imgs) - total, "images will be thrown away from the photomosaic")
    del imgs[total:]
    
    dest_img, img_keys = compute_blocks(colorspace, dest_img, imgs, grid)
    print("Computing cost matrix...")
    cols, cost = solve_lap(cdist(img_keys, dest_img, metric=metric).astype(ctype), v)

    print("Time taken: {}s".format((np.round(time.time() - t, 2))))
    return grid, [imgs[i] for i in cols]


def calc_col_dup(dest_img: np.ndarray, imgs: List[np.ndarray], max_width: int,
                 colorspace: str, metric: str, lower_thresh=None,
                 background=None, freq_mul=1, randomize=True) -> Tuple[Tuple[int, int], List[np.ndarray]]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels that constitute 
    of the salient objects of the target image, given that every image could be used arbitrary amount of times
    """
    t = time.time()
    
    # Because we don't have a fixed total amount of images as we can used a single image
    # for arbitrary amount of times, we need user to specify the maximum width in order to determine the grid size.
    rh, rw, _ = dest_img.shape
    rh = round(rh * max_width / rw)
    grid = (max_width, rh)
    print("Calculated grid size based on the aspect ratio of the image provided:", grid)

    salient = lower_thresh is not None and background is not None
    if salient:
        block_size = min(dest_img.shape[1] // grid[0], dest_img.shape[0] // grid[1])
        # we resize dest_img here so that the compute_block* functions can infer correct grid size from block_size
        dest_img = cv2.resize(dest_img, (grid[0] * block_size, grid[1] * block_size))
        _, thresh_map = cv2.saliency.StaticSaliencyFineGrained_create().computeSaliency((dest_img * 255).astype(np.uint8))

        ridx, cidx, thresh_map = compute_block_map(thresh_map, block_size, lower_thresh)
        dest_img, img_keys, bg_img = compute_blocks_salient(colorspace, dest_img, imgs, block_size, ridx, cidx, thresh_map, lower_thresh, background)
    else:
        dest_img, img_keys = compute_blocks(colorspace, dest_img, imgs, grid)

    print("Computing assignments...")
    img_keys = np.asarray(img_keys)
    assignment = np.full(dest_img.shape[0], -1, dtype=np.int32)
    _indices = np.arange(0, dest_img.shape[0], 1, dtype=np.int32)
    if randomize:
        np.random.shuffle(_indices)
    indices_freq = np.zeros(len(img_keys), dtype=np.float64)

    dist_mat = cdist(img_keys, dest_img, metric=metric).transpose().copy()
    if freq_mul > 0:
        for i in tqdm(_indices, desc="[Computing assignments]", unit="pixel", unit_divisor=1000, unit_scale=True,
                        ncols=pbar_ncols):
            # Compute the distance between the current pixel and each image in the set
            ranks = rankdata(dist_mat[i, :], "average")
            ranks += indices_freq
            idx = np.argmin(ranks)

            # Find the index of the image which best approximates the current pixel
            assignment[i] = idx
            indices_freq[idx] += freq_mul
    else:
        np.argmin(dist_mat, axis=1, out=assignment)
    print("Total assignment cost:", dist_mat[assignment, :].sum())

    if salient:
        imgs.append(bg_img)
        full_assignment = np.full(grid[::-1], len(imgs) - 1, dtype=np.int32)
        full_assignment[ridx, cidx] = assignment
        assignment = full_assignment.flatten()
    
    print("Time taken: {}s".format(np.round(time.time() - t, 2)))
    return grid, [imgs[i] for i in assignment]


def imwrite(filename: str, img: np.ndarray) -> None:
    ext = os.path.splitext(filename)[1]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
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

    with mp.Pool(max(1, num_process)) as pool:
        func = read_img_center if flag == "center" else read_img_other
        return [
            r for r in tqdm(
                pool.imap_unordered(func, zip(files, itertools.repeat(img_size, len(files))), chunksize=64), 
                total=len(files), desc="[Reading files]", unit="file", ncols=pbar_ncols) if r is not None
        ]


# this imread method can read images whose path contain unicode characters
def imread(filename: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    img *= 1 / 255.0
    return img


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
        return cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    except:
        return None


def unfair_exp_mat(dest_img, args, imgs):
    import matplotlib.pyplot as plt
    all_colorspaces = PARAMS.colorspace.choices
    all_freqs = np.zeros(6, dtype=np.float64)
    all_freqs[1:] = np.logspace(-2, 2, 5)
    grid, sorted_imgs, _ = calc_col_dup(dest_img, imgs, max_width=args.max_width, colorspace=all_colorspaces[0], freq_mul=all_freqs[0])
    img_shape = make_collage(grid, sorted_imgs, args.rev_row).shape
    grid_img = np.zeros((img_shape[0] * len(all_colorspaces), img_shape[1] * len(all_freqs), 3), dtype=np.uint8)

    pbar = tqdm(desc="[Experimenting]", total=len(all_freqs) * len(all_colorspaces), unit="exps")
    with con.ProcessPoolExecutor(4) as pool:
        futures = [
            [pool.submit(calc_col_dup, dest_img, imgs, max_width=args.max_width, colorspace=colorspace, metric=args.metric, freq_mul=freq) 
                for freq in all_freqs] 
                    for colorspace in all_colorspaces
        ]
        for i in range(len(all_colorspaces)):
            for j in range(len(all_freqs)):
                grid, sorted_imgs = futures[i][j].result()
                grid_img[i * img_shape[0]:(i+1)*img_shape[0], j*img_shape[1]:(j+1)*img_shape[1], :] = make_collage(grid, sorted_imgs, args.rev_row)
                pbar.update()
        pbar.refresh()
        plt.figure()
        plt.imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
        plt.yticks(np.arange(0, grid_img.shape[0], img_shape[0]) + img_shape[0] / 2, all_colorspaces)
        plt.xticks(np.arange(0, grid_img.shape[1], img_shape[1]) + img_shape[1] / 2, all_freqs)
        plt.show()


def unfair_exp(dest_img, args, imgs):
    import matplotlib.pyplot as plt
    all_colorspaces = PARAMS.colorspace.choices
    all_freqs = np.zeros(6, dtype=np.float64)
    all_freqs[1:] = np.logspace(-2, 2, 5)

    pbar = tqdm(desc="[Experimenting]", total=len(all_freqs) + len(all_colorspaces) + 1, unit="exps")

    with con.ProcessPoolExecutor(4) as pool:
        futures1 = [pool.submit(calc_col_dup, dest_img, imgs, max_width=args.max_width, colorspace=colorspace, metric=args.metric, freq_mul=1.0) 
                        for colorspace in all_colorspaces]
        futures2 = [pool.submit(calc_col_dup, dest_img, imgs, max_width=args.max_width, colorspace="bgr", metric=args.metric, freq_mul=freq) 
                        for freq in all_freqs]
        futures2.append(pool.submit(calc_col_even, dest_img, imgs, grid=futures2[-1].result()[0]))
        
        def collect_imgs(fname, params, futures, fs):
            result_imgs = []
            for i in range(len(params)):
                grid, sorted_imgs = futures[i].result()
                result_imgs.append(make_collage(grid, sorted_imgs, args.rev_row))
                pbar.update()
            
            plt.figure(figsize=(len(params) * 10, 12))
            plt.imshow(cv2.cvtColor(np.hstack(result_imgs), cv2.COLOR_BGR2RGB))
            grid_width = result_imgs[0].shape[1]
            plt.xticks(np.arange(0, grid_width * len(result_imgs), grid_width) + grid_width / 2, params, fontsize=fs)
            plt.yticks([], [])
            plt.subplots_adjust(left=0.005, right=0.995)
            plt.savefig(f"{fname}.png", dpi=100)
            # plt.xlabel(xlabel)

        collect_imgs("colorspace", [c.upper() for c in all_colorspaces], futures1, 36)
        collect_imgs("fairness", [f"$\lambda = {c}$" for c in all_freqs] + ["Fair"], futures2, 20)
        pbar.refresh()
        plt.show()


def sort_exp(args, imgs):
    pool = con.ProcessPoolExecutor(4)
    futures = {}

    for sort_method in PARAMS.sort.choices:
        futures[pool.submit(sort_collage, imgs, args.ratio, sort_method, args.rev_sort)] = sort_method

    for future in tqdm(con.as_completed(futures.keys()), total=len(PARAMS.sort.choices),
                        desc="[Experimenting]", unit="exps"):
        grid, sorted_imgs = future.result()
        combined_img = make_collage(grid, sorted_imgs, args.rev_row)
        save_img(combined_img, args.out, futures[future])


def main(args):
    if not args.verbose:
        sys.stdout = open(os.devnull, "w")

    if len(args.out) > 0:
        folder, file_name = os.path.split(args.out)
        if len(folder) > 0:
            assert os.path.isdir(folder), "The output path {} does not exist!".format(folder)
        # ext = os.path.splitext(file_name)[-1]
        # assert ext.lower() == ".jpg" or ext.lower() == ".png", "The file extension must be .jpg or .png"

    imgs = read_images(args.path, (args.size, args.size), args.recursive, args.num_process)
    if len(args.dest_img) == 0:
        if args.exp:
            sort_exp(args, imgs)
        else:
            grid, sorted_imgs = sort_collage(imgs, args.ratio, args.sort, args.rev_sort)
            save_img(make_collage(grid, sorted_imgs, args.rev_row), args.out, "")
        return

    assert os.path.isfile(args.dest_img)
    dest_img = imread(args.dest_img)

    if args.exp:
        assert not args.salient
        assert args.unfair
        unfair_exp(dest_img, args, imgs)        
        return
    
    if args.salient:
        if args.unfair:
            grid, sorted_imgs = calc_col_dup(
                dest_img, imgs, args.max_width, args.colorspace, args.metric,
                args.lower_thresh, args.background)
        else:
            grid, sorted_imgs = calc_salient_col_even(
                dest_img, imgs, args.dup, args.colorspace, args.ctype,
                args.metric, args.lower_thresh, args.background)
    else:
        if args.unfair:
            grid, sorted_imgs = calc_col_dup(
                dest_img, imgs, args.max_width, args.colorspace, args.metric)
        else:
            grid, sorted_imgs = calc_col_even(
                dest_img, imgs, args.dup, args.colorspace, args.ctype, args.metric)
    save_img(make_collage(grid, sorted_imgs, args.rev_row), args.out, "")


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for arg_name, data in PARAMS.__dict__.items():
        if arg_name.startswith("__"):
            continue
        
        arg_name = "--" + arg_name
        if data.type == bool:
            assert data.default == False
            parser.add_argument(arg_name, action="store_true", help=data.help)
            continue
        
        parser.add_argument(arg_name, type=data.type, default=data.default, help=data.help, choices=data.choices, nargs=data.nargs)
    parser.add_argument("--exp", action="store_true", help="Do experiments (for testing only)")
    main(parser.parse_args())
