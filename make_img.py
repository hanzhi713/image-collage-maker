import os
import sys
import time
import math
import random
import argparse
import itertools
import traceback
import multiprocessing as mp
from fractions import Fraction
from typing import Any, Callable, List, Tuple, Type
from collections import defaultdict

from io_utils import stdout_redirector, JVOutWrapper

import cv2
import imagesize
import numpy as np
cp = np
from tqdm import tqdm
from lapjv import lapjv

Grid = Tuple[int, int] # grid size = (width, height)
BackgroundRGB = Tuple[int, int, int]

if mp.current_process().name != "MainProcess":
    sys.stdout = open(os.devnull, "w")
    sys.stderr = sys.stdout

pbar_ncols = None
LIMIT = 2**32


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
    num_process = _PARAMETER(type=int, default=mp.cpu_count() // 2, help="Number of processes to use for parallelizable operations")
    out = _PARAMETER(default="result.png", type=str, help="The filename of the output collage/photomosaic")
    size = _PARAMETER(type=int, nargs="+", default=(50,), 
        help="Width and height of each tile in pixels in the resulting collage/photomosaic. "
             "If two numbers are specified, they are treated as width and height. "
             "If one number is specified, the number is treated as the width"
             "and the height is inferred from the aspect ratios of the images provided. ")
    quiet = _PARAMETER(type=bool, default=False, help="Do not print progress message to console")
    auto_rotate = _PARAMETER(type=int, default=0, choices=[-1, 0, 1],
        help="Options to auto rotate tiles to best match the specified tile size. 0: do not auto rotate. "
             "1: attempt to rotate counterclockwise by 90 degrees. -1: attempt to rotate clockwise by 90 degrees")
    resize_opt = _PARAMETER(type=str, default="center", choices=["center", "stretch"], 
        help="How to resize each tile so they become square images. "
             "Center: crop a square in the center. Stretch: stretch the tile")
    gpu = _PARAMETER(type=bool, default=False, 
        help="Use GPU acceleration. Requires cupy to be installed and a capable GPU. Note that USUALLY this is useful when you: "
             "1. only have few cpu cores, and "
             "2. have a lot of tiles (typically > 10000) "
             "3. and are using the unfair mode. "
             "Also note: enabling GPU acceleration will disable multiprocessing on CPU for videos"
    )
    mem_limit = _PARAMETER(type=int, default=4096, 
        help="The APPROXIMATE memory limit in MB when computing a photomosaic in unfair mode. Applicable both CPU and GPU computing. "
             "If you run into memory issues when using GPU, try reduce this memory limit")
    tile_info_out = _PARAMETER(type=str, default="",
        help="Path to save the list of tile filenames for the collage/photomosaic. If empty, it will not be saved.")
    
    # ---------------- sort collage options ------------------
    ratio = _PARAMETER(type=int, default=(16, 9), help="Aspect ratio of the output image", nargs=2)
    sort = _PARAMETER(type=str, default="bgr_sum", help="Sort method to use", choices=[
        "none", "bgr_sum", "av_hue", "av_sat", "av_lum", "rand"
    ])
    rev_row = _PARAMETER(type=bool, default=False, help="Whether to use the S-shaped alignment.")
    rev_sort = _PARAMETER(type=bool, default=False, help="Sort in the reverse direction.")
    
    # ---------------- photomosaic common options ------------------
    dest_img = _PARAMETER(type=str, default="", help="The path to the destination image that you want to build a photomosaic for")
    colorspace = _PARAMETER(type=str, default="lab", choices=["hsv", "hsl", "bgr", "lab", "luv"], 
        help="The colorspace used to calculate the metric")
    metric = _PARAMETER(type=str, default="euclidean", choices=["euclidean", "cityblock", "chebyshev", "cosine"], 
        help="Distance metric used when evaluating the distance between two color vectors")
    
    # ---- unfair tile assignment options -----
    unfair = _PARAMETER(type=bool, default=False, 
        help="Whether to allow each tile to be used different amount of times (unfair tile usage). ")
    max_width = _PARAMETER(type=int, default=80, 
        help="Maximum width of the collage. This option is only valid if unfair option is enabled")    
    freq_mul = _PARAMETER(type=float, default=0.0, 
        help="Frequency multiplier to balance tile fairless and mosaic quality. Minimum: 0. "
             "More weight will be put on tile fairness when this number increases.")
    deterministic = _PARAMETER(type=bool, default=False, 
        help="Do not randomize the tiles. This option is only valid if unfair option is enabled")

    # --- fair tile assignment options ---
    dup = _PARAMETER(type=float, default=1, 
        help="If a positive integer: duplicate the set of tiles by how many times. Can be a fraction")

    # ---- saliency detection options ---
    salient = _PARAMETER(type=bool, default=False, help="Make photomosaic for salient objects only")
    lower_thresh = _PARAMETER(type=float, default=0.5, 
        help="The threshold for saliency detection, between 0.0 (no object area = blank) and 1.0 (maximum object area = original image)")
    background = _PARAMETER(nargs=3, type=int, default=(255, 255, 255), 
        help="Background color in RGB for non salient part of the image")

    # ---- blending options ---
    blending = _PARAMETER(type=str, default="alpha", choices=["alpha", "brightness"], 
        help="The types of blending used. alpha: alpha (transparency) blending. Brightness: blending of brightness (lightness) channel in the HSL colorspace")
    blending_level = _PARAMETER(type=float, default=0.0, 
        help="Level of blending, between 0.0 (no blending) and 1.0 (maximum blending). Default is no blending")

    video = _PARAMETER(type=bool, default=False, help="Make a photomosaic video from dest_img which is assumed to be a video")
    skip_frame = _PARAMETER(type=int, default=1, help="Make a photomosaic every this number of frames")

# https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
class InfoArray(np.ndarray):
    def __new__(cls, input_array, info=''):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(InfoArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.info,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.info = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(InfoArray, self).__setstate__(state[0:-1])


ImgList = List[InfoArray]
cupy_available = False


def fast_sq_euclidean(Asq, Bsq, AB):
    AB *= -2
    AB += Asq
    AB += Bsq
    return AB


def fast_cityblock(A, B, axis, out):
    Z = A - B
    np.abs(Z, out=Z)
    return np.sum(Z, axis=axis, out=out)


def fast_chebyshev(A, B, axis, out):
    Z = A - B
    np.abs(Z, out=Z)
    return np.max(Z, axis=axis, out=out)


def to_cpu(X: np.ndarray) -> np.ndarray:
    return X.get() if cupy_available else X


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


lum_coeffs = np.array([0.241, 0.691, 0.068], dtype=np.float32)[np.newaxis, np.newaxis, :]


def av_lum(img) -> float:
    """
    compute the average luminosity
    """
    lum = img * lum_coeffs
    np.sqrt(lum, out=lum)
    return np.mean(lum)


def rand(img: np.ndarray) -> float:
    """
    generate a random number for each image
    """
    return random.random()


def calc_grid_size(rw: int, rh: int, num_imgs: int, shape: Tuple[int, int, int]) -> Grid:
    """
    :param rw: the width of the target image
    :param rh: the height of the target image
    :param num_imgs: number of images available
    :param shape: the shape of a tile
    :return: an optimal grid size
    """
    possible_wh = []
    th, tw, _ = shape
    for width in range(1, num_imgs):
        height = math.ceil(num_imgs / width)
        possible_wh.append((width * tw / (th * height), width, height))
    dest_ratio = rw / rh
    grid = min(possible_wh, key=lambda x: (x[0] - dest_ratio) ** 2)[1:]
    print("Calculated grid size based on the aspect ratio of the destination image:", grid)
    print(f"Collage size will be {grid[0] * tw}x{grid[1] * th}. ")
    return grid


def make_collage(grid: Grid, sorted_imgs: ImgList, rev=False, file=None) -> np.ndarray:
    """
    :param grid: grid size
    :param sorted_imgs: list of images sorted in correct position
    :param rev: whether to have opposite alignment for consecutive rows
    :return: a collage
    """
    print("Aligning images on the grid...", file=file)
    total = np.prod(grid)
    if len(sorted_imgs) < total:
        diff = total - len(sorted_imgs)
        print(f"Note: {diff} white tiles will be added to the grid.")
        sorted_imgs.extend([get_background_tile(sorted_imgs[0].shape, (255, 255, 255))] * diff)
    elif len(sorted_imgs) > total:
        print(f"Note: {len(sorted_imgs) - total} tiles will be dropped from the grid.")
        del sorted_imgs[total:]

    combined_img = np.asarray([img.view(np.float32) for img in sorted_imgs])
    combined_img.shape = (*grid[::-1], *sorted_imgs[0].shape)
    if rev:
        combined_img[1::2] = combined_img[1::2, ::-1]
    combined_img = combined_img.transpose((0, 2, 1, 3, 4))
    combined_img = combined_img.reshape(np.prod(combined_img.shape[:2]), -1, 3)
    return combined_img, f"Grid dimension: {grid}\n" + '\n'.join([img.info for img in sorted_imgs])


def alpha_blend(combined_img: np.ndarray, dest_img: np.ndarray, alpha=0.9):
    dest_img = cv2.resize(dest_img, combined_img.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    dest_img *= 1 - alpha
    combined_img = combined_img * alpha # copy
    combined_img += dest_img
    return combined_img


def brightness_blend(combined_img: np.ndarray, dest_img: np.ndarray, alpha=0.9):
    """
    blend the 2 imgs in the lightness channel (L in HSL)
    """
    dest_img = cv2.resize(dest_img, combined_img.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    cv2.cvtColor(dest_img, cv2.COLOR_BGR2HLS, dst=dest_img)
    dest_img[:, :, 1] *= 1 - alpha
    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2HLS)
    combined_img[:, :, 1] *= alpha
    combined_img[:, :, 1] += dest_img[:, :, 1]
    cv2.cvtColor(combined_img, cv2.COLOR_HLS2BGR, dst=combined_img)
    return combined_img


def sort_collage(imgs: ImgList, ratio: Grid, sort_method="pca_lab", rev_sort=False) -> Tuple[Grid, np.ndarray]:
    """
    :param imgs: list of images
    :param ratio: The aspect ratio of the collage
    :param sort_method:
    :param rev_sort: whether to reverse the sorted array
    :return: [calculated grid size, sorted image array]
    """
    t = time.time()
    grid = calc_grid_size(ratio[0], ratio[1], len(imgs), imgs[0].shape)
    total = np.prod(grid)
    if len(imgs) < total:
        diff = total - len(imgs)
        print(f"Note: {diff} white tiles will be added to the sorted collage.")
        imgs = imgs + [get_background_tile(imgs[0].shape, (255, 255, 255))] * diff

    if sort_method == "none":
        return grid, imgs

    print("Sorting images...")    
    sort_function = eval(sort_method)
    indices = np.array(list(map(sort_function, imgs))).argsort()
    if rev_sort:
        indices = indices[::-1]
    print("Time taken: {}s".format(np.round(time.time() - t, 2)))
    return grid, [imgs[i] for i in indices]


def solve_lap(cost_matrix: np.ndarray, v=-1):
    if v == -1:
        v = sys.__stderr__
    """
    solve the linear sum assignment (LAP) problem with progress info
    """
    print("Computing optimal assignment on a {}x{} matrix...".format(cost_matrix.shape[0], cost_matrix.shape[1]))
    wrapper = JVOutWrapper(v, pbar_ncols)
    with stdout_redirector(wrapper):
        _, cols, cost = lapjv(cost_matrix, verbose=1)
        wrapper.finish()
    cost = cost[0]
    print("Total assignment cost:", cost)
    return cols


def solve_lap_greedy(cost_matrix: np.ndarray, v=None):
    assert cost_matrix.shape[0] == cost_matrix.shape[1]

    print("Computing greedy assignment on a {}x{} matrix...".format(cost_matrix.shape[0], cost_matrix.shape[1]))
    row_idx, col_idx = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
    cost = 0
    row_assigned = np.full(cost_matrix.shape[0], -1, dtype=np.int32)
    col_assigned = np.full(cost_matrix.shape[0], -1, dtype=np.int32)
    pbar = tqdm(ncols=pbar_ncols, total=cost_matrix.shape[0])
    for ridx, cidx in zip(row_idx, col_idx):
        if row_assigned[ridx] == -1 and col_assigned[cidx] == -1:
            row_assigned[ridx] = cidx
            col_assigned[cidx] = ridx
            cost += cost_matrix[ridx, cidx]

            pbar.update()
            if pbar.n == pbar.total:
                break
    pbar.close()
    print("Total assignment cost:", cost)
    return col_assigned


def compute_block_map(thresh_map: np.ndarray, block_width: int, block_height: int, lower_thresh: int):
    """
    Find the indices of the blocks that contain salient pixels according to the thresh_map

    returns [row indices, column indices, resized threshold map] of sizes [(N,), (N,), (W x H)]
    """
    height, width = thresh_map.shape
    dst_size = (width - width % block_width, height - height % block_height)
    if thresh_map.shape[::-1] != dst_size:
        thresh_map = cv2.resize(thresh_map, dst_size, interpolation=cv2.INTER_AREA)
    row_idx, col_idx = np.nonzero(thresh_map.reshape(
        dst_size[1] // block_height, block_height, dst_size[0] // block_width, block_width).max(axis=(1, 3)) >= lower_thresh
    )
    return row_idx, col_idx, thresh_map


def get_background_tile(shape: Tuple[int], background: BackgroundRGB):
    bg = np.asarray(background[::-1], dtype=np.float32)
    bg *= 1 / 255.0
    return InfoArray(np.full(shape, bg, dtype=np.float32), f"background-{'-'.join(str(a) for a in background)}")


def dup_to_meet_total(imgs: ImgList, total: int):
    """
    note that this function modifies imgs in place
    """
    orig_len = len(imgs)
    if total < orig_len:
        print(f"{total} tiles will be used 1 time. {orig_len - total}/{orig_len} tiles will not be used. ")
        del imgs[total:]
        return imgs
    
    full_count = total // orig_len
    remaining = total % orig_len

    imgs *= full_count
    if remaining > 0:
        print(f"{orig_len - remaining} tiles will be used {full_count} times. {remaining} tiles will be used {full_count + 1} times. Total tiles: {orig_len}.")
        imgs.extend(imgs[:remaining])
    else:
        print(f"Total tiles: {orig_len}. All of them will be used {full_count} times.")
    return imgs


def _cosine(A, B):
    return 1 - cp.inner(A / cp.linalg.norm(A, axis=1, keepdims=True), B)


def _euclidean(A, B, BsqT):
    Asq = cp.sum(A**2, axis=1, keepdims=True)
    return fast_sq_euclidean(Asq, BsqT, A.dot(B.T))


def _other(A, B, dist_func, row_stride):
    total = A.shape[0]
    dist_mat = cp.empty((total, B.shape[1]), dtype=cp.float32)
    i = 0
    while i < total - row_stride:
        next_i = i + row_stride
        dist_func(A[i:next_i, cp.newaxis, :], B, out=dist_mat[i:next_i], axis=2)
        i = next_i
    if i < total:
        dist_func(A[i:, cp.newaxis, :], B, out=dist_mat[i:], axis=2)
    return dist_mat


class CachedCDist:
    def __init__(self, metric: str, B: np.ndarray):
        """
        Simple implementation of scipy.spatial.distance.cdist
        """
        if metric == "cosine":
            self.args = [B / cp.linalg.norm(B, axis=1, keepdims=True)]
            self.func = _cosine
        elif metric == "euclidean":
            self.args = [B, cp.sum(B**2, axis=1, keepdims=True).T]
            self.func = _euclidean
        else:
            row_stride = LIMIT // (B.size * 4)
            B = B[cp.newaxis]
            if metric == "cityblock":
                self.args = [B, fast_cityblock, row_stride]
            elif metric == "chebyshev":
                self.args = [B, fast_chebyshev, row_stride]
            else:
                raise ValueError(f"invalid metric {metric}")
            self.func = _other

    def __call__(self, A: np.ndarray) -> np.ndarray:
        return self.func(A, *self.args)


class MosaicCommon:
    def __init__(self, imgs: ImgList, colorspace="lab") -> None:
        self.imgs = imgs
        self.normalize_first = False
        if colorspace == "bgr":
            self.flag = None
        elif colorspace == "hsv":
            self.flag = cv2.COLOR_BGR2HSV
            self.normalize_first = True
        elif colorspace == "hsl":
            self.flag = cv2.COLOR_BGR2HLS
            self.normalize_first = True
        elif colorspace == "lab":
            self.flag = cv2.COLOR_BGR2LAB
        elif colorspace == "luv":
            self.flag = cv2.COLOR_BGR2LUV
        else:
            raise ValueError("Unknown colorspace " + colorspace)

    def combine_imgs(self):
        self.combined_img = np.asarray([img.view(np.float32) for img in self.imgs])

    def make_photomosaic(self, assignment: np.ndarray):
        grid_assignment = assignment.reshape(self.grid[::-1])
        combined_img = self.combined_img[grid_assignment, :, : , :].transpose((0, 2, 1, 3, 4))
        return combined_img.reshape(np.prod(combined_img.shape[:2]), -1, 3), \
               f"Grid dimension: {self.grid}\n" + '\n'.join([self.imgs[i].info for i in assignment])

    def convert_colorspace(self, img: np.ndarray):
        if self.flag is None:
            return
        cv2.cvtColor(img, self.flag, dst=img)
        if self.normalize_first:
            # for hsv/hsl, h is in range 0~360 while other channels are in range 0~1
            # need to normalize
            img[:, :, 0] *= 1 / 360.0
        
    def compute_block_size(self, dest_shape: Tuple[int, int, int], grid: Grid):
        self.grid = grid
        self.block_height = round(dest_shape[0] / grid[1])
        self.block_width = round(dest_shape[1] / grid[0])
        th, tw, _ = self.imgs[0].shape

        if self.block_width > tw or self.block_height > th:
            m = max(tw / self.block_width, th / self.block_height)
            self.block_width = math.floor(self.block_width * m)
            self.block_height = math.floor(self.block_height * m)
        self.flat_block_size = self.block_width * self.block_height * 3
        print("Block size:", (self.block_width, self.block_height))

        self.target_sz = (grid[0] * self.block_width, grid[1] * self.block_height)
        print(f"Resizing dest image from {dest_shape[1]}x{dest_shape[0]} to {self.target_sz[0]}x{self.target_sz[1]}")

    def imgs_to_flat_blocks(self, metric: str):
        img_keys = np.zeros((len(self.imgs), self.block_height, self.block_width, 3), dtype=np.float32)
        for i in range(len(self.imgs)):
            cv2.resize(self.imgs[i], (self.block_width, self.block_height), dst=img_keys[i], interpolation=cv2.INTER_AREA)
        img_keys.shape = (-1, self.block_width, 3)
        self.convert_colorspace(img_keys)
        img_keys.shape = (-1, self.flat_block_size)
        img_keys = cp.asarray(img_keys)
        self.cdist = CachedCDist(metric, img_keys)
        return img_keys

    def dest_to_flat_blocks(self, dest_img: np.ndarray):
        dest_img = cv2.resize(dest_img, self.target_sz, interpolation=cv2.INTER_LINEAR)
        self.convert_colorspace(dest_img)
        dest_img = cp.asarray(dest_img)
        dest_img.shape = (self.grid[1], self.block_height, self.grid[0], self.block_width, 3)
        return dest_img.transpose((0, 2, 1, 3, 4)).reshape(-1, self.flat_block_size)

    def dest_to_flat_blocks_mask(self, dest_img: np.ndarray, lower_thresh: int, ridx: np.ndarray, cidx: np.ndarray, thresh_map: np.ndarray):
        dest_img[thresh_map < lower_thresh] = self.imgs[-1][0, 0, :]
        self.convert_colorspace(dest_img)
        dest_img.shape = (self.grid[1], self.block_height, self.grid[0], self.block_width, 3)
        dest_img = dest_img[ridx, :, cidx, :, :]
        dest_img.shape = (-1, self.flat_block_size)
        print(f"Salient blocks/total blocks = {len(ridx)}/{np.prod(self.grid)}")
        return cp.asarray(dest_img)


def calc_salient_col_even(dest_img: np.ndarray, imgs: ImgList, dup=1, colorspace="lab", 
                          metric="euclidean", lower_thresh=0.5, background=(255, 255, 255), v=None) -> Tuple[Grid, ImgList]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels constitute of salient objects of the
    target image, with the restriction that every image should be used the same amount of times

    non salient part of the target image is filled with background color=background
    """
    t = time.time()
    print("Duplicating {} times".format(dup))
    height, width, _ = dest_img.shape

    # this is just the initial (minimum) grid size
    total = round(len(imgs) * dup)
    grid = calc_grid_size(width, height, total, imgs[0].shape)
    _, orig_thresh_map = cv2.saliency.StaticSaliencyFineGrained_create().computeSaliency((dest_img * 255).astype(np.uint8))
    
    bh_f = height / grid[1]
    bw_f = width / grid[0]
    # DDA-like algorithm to decrease block size while preserving aspect ratio
    if bw_f > bh_f:
        bw_delta = 1
        bh_delta = bh_f / bw_f
    else:
        bh_delta = 1
        bw_delta = bh_f / bw_f
    
    while True:
        block_width = int(bw_f)
        block_height = int(bh_f)
        ridx, cidx, thresh_map = compute_block_map(orig_thresh_map, block_width, block_height, lower_thresh)
        if len(ridx) >= total:
            break
        bw_f -= bw_delta
        bh_f -= bh_delta
        assert bw_f > 0 and bh_f > 0, "Salient area is too small to put down all tiles. Please try to increase the saliency threshold."

    imgs = dup_to_meet_total(imgs.copy(), len(ridx))

    mos = MosaicCommon(imgs, colorspace)
    mos.block_width = block_width
    mos.block_height = block_height
    mos.flat_block_size = block_width * block_height * 3
    mos.grid = (thresh_map.shape[1] // block_width, thresh_map.shape[0] // block_height)

    print("Block size:", (block_width, block_height))
    print("Grid size:", mos.grid)

    mos.imgs_to_flat_blocks(metric)
    mos.imgs.append(get_background_tile(imgs[0].shape, background))
    mos.combine_imgs()
    dest_img = cv2.resize(dest_img, thresh_map.shape[::-1], interpolation=cv2.INTER_AREA)
    dest_img = mos.dest_to_flat_blocks_mask(dest_img, lower_thresh, ridx, cidx, thresh_map)
    
    cols = solve_lap(to_cpu(mos.cdist(dest_img).T), v)
    assignment = np.full(mos.grid[::-1], len(mos.imgs) - 1, dtype=np.int32)
    assignment[ridx, cidx] = cols
    print("Time taken: {}s".format((np.round(time.time() - t, 2))))
    return mos.make_photomosaic(assignment)


class MosaicFairSalient:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
    
    def process_dest_img(self, dest_img: np.ndarray):
        return calc_salient_col_even(dest_img, *self.args[1:], **self.kwargs)


class MosaicFair(MosaicCommon):
    def __init__(self, dest_shape: Tuple[int, int, int], imgs: ImgList, dup=1, colorspace="lab", 
            metric="euclidean", grid=None) -> None:
        """
        Compute the optimal assignment between the set of images provided and the set of pixels of the target image,
        with the restriction that every image should be used the same amount of times
        """
        if grid is not None:
            print("Use the provided grid size:", grid)
            dup = np.prod(grid) // len(imgs) + 1
        else:
            # Compute the grid size based on the number images that we have
            grid = calc_grid_size(dest_shape[1], dest_shape[0], round(len(imgs) * dup), imgs[0].shape)
        total = np.prod(grid)
        imgs = dup_to_meet_total(imgs.copy(), total)
        
        if total > 10000:
            print("Warning: this may take longer than 5 minutes to compute")
    
        super().__init__(imgs, colorspace)
        self.compute_block_size(dest_shape, grid)
        self.imgs_to_flat_blocks(metric)
        self.combine_imgs()

    def process_dest_img(self, dest_img: np.ndarray, file=None):
        dest_img = self.dest_to_flat_blocks(dest_img)
        cols = solve_lap(to_cpu(self.cdist(dest_img).T), file)
        return self.make_photomosaic(cols)


class MosaicUnfair(MosaicCommon):
    def __init__(self, dest_shape: Tuple[int, int, int], imgs: ImgList, max_width, colorspace, metric, lower_thresh, background, freq_mul, randomize) -> None:
        # Because we don't have a fixed total amount of images as we can used a single image
        # for arbitrary amount of times, we need user to specify the maximum width in order to determine the grid size.
        dh, dw, _ = dest_shape
        th, tw, _ = imgs[0].shape
        grid = (max_width, round(dh * (max_width * tw / dw) / th))
        print("Calculated grid size based on the aspect ratio of the image provided:", grid)
        print("Collage size:", (grid[0] * tw, grid[1] * th))

        super().__init__(imgs, colorspace)
        self.compute_block_size(dest_shape, grid)
        img_keys = self.imgs_to_flat_blocks(metric)

        # number of rows in the cost matrix
        # note here we compute the cost matrix chunk by chunk to limit memory usage
        # a bit like sklearn.metrics.pairwise_distances_chunked
        num_rows = np.prod(grid)
        num_cols = img_keys.shape[0]
        print(f"Distance matrix size: {(num_rows, num_cols)} = {num_rows * num_cols * 4 / 2**20}MB")
        self.row_stride = (LIMIT - (img_keys.size + num_rows * (1 + self.flat_block_size)) * 4) // (num_cols * 4)
        if self.row_stride >= num_rows:
            print("No chunking will be performed on the distance matrix calculation")
        else:
            print(f"Chunk size: {self.row_stride*num_cols* 4 / 2**20}MB | {self.row_stride}/{num_rows}")

        if freq_mul > 0:
            self.row_stride //= 16
            self.indices_freq = cp.empty(num_cols, dtype=cp.float32)
            self.row_range = cp.arange(0, self.row_stride, dtype=cp.int32)[:, cp.newaxis]
            self.temp = cp.arange(0, num_cols, dtype=cp.float32)
        else:
            self.row_stride //= 4

        self.freq_mul = freq_mul
        self.lower_thresh = lower_thresh
        self.randomize = randomize
        
        self.saliency = None
        if lower_thresh is not None and background is not None:
            self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            self.imgs = self.imgs.copy()
            self.imgs.append(get_background_tile(imgs[0].shape, background))
        self.combine_imgs()

    def process_dest_img(self, dest_img: np.ndarray, file=None):
        if self.saliency is not None:
            dest_img = cv2.resize(dest_img, self.target_sz, interpolation=cv2.INTER_LINEAR)
            _, thresh_map = self.saliency.computeSaliency((dest_img * 255).astype(np.uint8))
            ridx, cidx, thresh_map = compute_block_map(thresh_map, self.block_width, self.block_height, self.lower_thresh)
            dest_img = self.dest_to_flat_blocks_mask(dest_img, self.lower_thresh, ridx, cidx, thresh_map)
        else:
            dest_img = self.dest_to_flat_blocks(dest_img)

        total = dest_img.shape[0]
        assignment = cp.empty(total, dtype=cp.int32)
        pbar = tqdm(desc="[Computing assignments]", total=total, ncols=pbar_ncols, file=file)
        i = 0
        row_stride = self.row_stride
        if self.freq_mul > 0:
            _indices = np.arange(0, total, dtype=np.int32)
            if self.randomize:
                np.random.shuffle(_indices)
            dest_img = dest_img[_indices] # reorder the rows of dest img
            indices_freq = self.indices_freq
            indices_freq.fill(0.0)
            freq_mul = self.freq_mul
            while i < total - row_stride:
                dist_mat = self.cdist(dest_img[i:i+row_stride])
                dist_mat[self.row_range, cp.argsort(dist_mat, axis=1)] = self.temp
                j = 0
                while j < row_stride:
                    row = dist_mat[j, :]
                    row += indices_freq
                    idx = cp.argmin(row)
                    assignment[i] = idx
                    indices_freq[idx] += freq_mul
                    i += 1
                    j += 1
                    pbar.update()
            if i < total:
                dist_mat = self.cdist(dest_img[i:])
                dist_mat[self.row_range[:total - i], cp.argsort(dist_mat, axis=1)] = self.temp
                j = 0
                while i < total:
                    row = dist_mat[j, :]
                    row += indices_freq
                    idx = cp.argmin(row)
                    assignment[i] = idx
                    indices_freq[idx] += freq_mul
                    i += 1
                    j += 1
                    pbar.update()
            assignment[_indices] = assignment.copy()
        else:
            while i < total - row_stride:
                next_i = i + row_stride
                dist_mat = self.cdist(dest_img[i:next_i])
                cp.argmin(dist_mat, axis=1, out=assignment[i:next_i])
                pbar.update(row_stride)
                i = next_i
            if i < total:
                dist_mat = self.cdist(dest_img[i:])
                cp.argmin(dist_mat, axis=1, out=assignment[i:])
                pbar.update(total - i)
        pbar.close()

        assignment = to_cpu(assignment)
        if self.saliency is not None:
            full_assignment = np.full(self.grid[::-1], len(self.imgs) - 1, dtype=np.int32)
            full_assignment[ridx, cidx] = assignment
            assignment = full_assignment
        return self.make_photomosaic(assignment)


def imwrite(filename: str, img: np.ndarray) -> None:
    ext = os.path.splitext(filename)[1]
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    result, n = cv2.imencode(ext, img)
    assert result, "Error saving the collage"
    n.tofile(filename)


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


def get_size(img):
    try:
        return imagesize.get(img)
    except:
        return -1, -1


def get_size_slow(img):
    try:
        return imread(img).shape[1::-1]
    except:
        return -1, -1


def infer_size(pool: Type[mp.Pool], files: List[str], infer_func: Callable[[str], Tuple[int, int]], i_type: str):
    sizes = defaultdict(int)
    for w, h in tqdm(pool.imap_unordered(infer_func, files, chunksize=64), 
        total=len(files), desc=f"[Inferring size ({i_type})]", ncols=pbar_ncols):
        if h == 0: # skip zero size images
            continue
        sizes[Fraction(w, h)] += 1
    if Fraction(-1, -1) in sizes:
        del sizes[Fraction(-1, -1)]
    sizes = [(args[1], args[0].numerator / args[0].denominator) for args in sizes.items()]
    sizes.sort()
    return sizes


def read_images(pic_path: str, img_size: List[int], recursive, pool: mp.Pool, flag="stretch", auto_rotate=0) -> ImgList:
    assert os.path.isdir(pic_path), "Directory " + pic_path + "is non-existent"
    files = []
    print("Scanning files...")
    for root, _, file_list in os.walk(pic_path):
        for f in file_list:
            files.append(os.path.join(root, f))
        if not recursive:
            break

    if len(img_size) == 1:
        sizes = infer_size(pool, files, get_size, "fast")
        if len(sizes) == 0:
            print("Warning: unable to infer image size through metadata. Will try reading the entire image (slow!)")
            sizes = infer_size(pool, files, get_size_slow, "slow")
            assert len(sizes) > 0, "Fail to infer size. All of your images are in an unsupported format!"

        # print("Aspect ratio (width / height, sorted by frequency) statistics:")
        # for freq, ratio in sizes:
        #     print(f"{ratio:6.4f}: {freq}")

        most_freq_ratio = 1 / sizes[-1][1]
        img_size = (img_size[0], round(img_size[0] * most_freq_ratio))
        print("Inferred tile size:", img_size)
    else:
        assert len(img_size) == 2
        img_size = (img_size[0], img_size[0])

    result = [
        r for r in tqdm(
            pool.imap_unordered(
                read_img_center if flag == "center" else read_img_other, 
                zip(files, itertools.repeat(img_size, len(files)), itertools.repeat(auto_rotate, len(files))), 
            chunksize=32), 
            total=len(files), desc="[Reading files]", unit="file", ncols=pbar_ncols) 
                if r is not None
    ]
    print(f"Read {len(result)} images. {len(files) - len(result)} files cannot be decode as images.")
    return result


def imread(filename: str) -> np.ndarray:
    """
    like cv2.imread, but can read images whose path contain unicode characters
    """
    img = cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return img
    img = img.astype(np.float32)
    img *= 1 / 255.0
    return img


def read_img_center(args: Tuple[str, Tuple[int, int], int]):
    # crop the largest square from the center of a non-square image
    img_file, img_size, rot = args
    img = imread(img_file)
    if img is None:
        return None
    
    ratio = img_size[0] / img_size[1]
    # rotate the image if possible to preserve more area
    h, w, _ = img.shape
    if rot != 0 and abs(h / w - ratio) < abs(w / h - ratio):
        img = np.rot90(img, k=rot)
        w, h = h, w
    
    cw = round(h * ratio) # cropped width
    ch = round(w / ratio) # cropped height
    assert cw <= w or ch <= h
    cond = cw > w or (ch <= h and (w - cw) * h > (h - ch) * w)
    if cond:
        img = img.transpose((1, 0, 2))
        w, h = h, w
        cw = ch
    
    margin = (w - cw) // 2
    add = (w - cw) % 2
    img = img[:, margin:w - margin + add, :]
    if cond:
        img = img.transpose((1, 0, 2))
    return InfoArray(cv2.resize(img, img_size, interpolation=cv2.INTER_AREA), img_file)


def read_img_other(args: Tuple[str, Tuple[int, int], int]):
    img_file, img_size, rot = args
    img = imread(img_file)
    if img is None:
        return img
    
    if rot != 0:
        ratio = img_size[0] / img_size[1]
        h, w, _ = img.shape
        if abs(h / w - ratio) < abs(w / h - ratio):
            img = np.rot90(img, k=rot)
    return InfoArray(cv2.resize(img, img_size, interpolation=cv2.INTER_AREA), img_file)


# pickleable helper classes for unfair exp
class _HelperChangeFreq:
    def __init__(self, dest_img: np.ndarray, mos: MosaicUnfair) -> None:
        self.mos = mos
        self.dest_img = dest_img

    def __call__(self, freq) -> Any:
        self.mos.freq_mul = freq
        return self.mos.process_dest_img(self.dest_img)

class _HelperChangeColorspace:
    def __init__(self, dest_img, *args) -> None:
        self.dest_img = dest_img
        self.args = list(args)

    def __call__(self, colorspace) -> Any:
        self.args[3] = colorspace
        return MosaicUnfair(*self.args).process_dest_img(self.dest_img)

def unfair_exp(dest_img: np.ndarray, args, imgs):
    import matplotlib.pyplot as plt
    all_colorspaces = PARAMS.colorspace.choices
    all_freqs = np.zeros(6, dtype=np.float64)
    all_freqs[1:] = np.logspace(-2, 2, 5)

    pbar = tqdm(desc="[Experimenting]", total=len(all_freqs) + len(all_colorspaces) + 1, unit="exps")
    mos_bgr = MosaicUnfair(dest_img.shape, imgs, args.max_width, "bgr", args.metric, None, None, 1.0, not args.deterministic)
    mos_fair = MosaicFair(dest_img.shape, imgs, colorspace="bgr", grid=mos_bgr.grid)
    change_cp = _HelperChangeColorspace(dest_img, dest_img.shape, imgs, args.max_width, None, args.metric, None, None, 1.0, not args.deterministic)
    change_freq = _HelperChangeFreq(dest_img, mos_bgr)

    with mp.Pool(4) as pool:
        futures1 = [pool.apply_async(change_cp, (colorspace,)) for colorspace in all_colorspaces]
        futures2 = [pool.apply_async(change_freq, (freq,)) for freq in all_freqs]
        futures2.append(pool.apply_async(mos_fair.process_dest_img, (dest_img,)))
        
        def collect_imgs(fname, params, futures, fs):
            result_imgs = []
            for i in range(len(params)):
                result_imgs.append(futures[i].get()[0])
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
        collect_imgs("fairness", [f"Frequency multiplier ($\lambda$) = ${c}$" for c in all_freqs] + ["Fair"], futures2, 20)
        pbar.refresh()
        # plt.show()


def sort_exp(pool, args, imgs):
    n = len(PARAMS.sort.choices)
    for sort_method, (grid, sorted_imgs) in zip(
        PARAMS.sort.choices, pool.starmap(sort_collage, 
            zip(itertools.repeat(imgs, n), 
            itertools.repeat(args.ratio, n), 
            PARAMS.sort.choices, 
            itertools.repeat(args.rev_sort, n))
        )):
        save_img(make_collage(grid, sorted_imgs, args.rev_row)[0], args.out, sort_method)


def frame_generator(ret, frame, dest_video, skip_frame):
    i = 0
    while ret:
        if i % skip_frame == 0:
            yield frame
        ret, frame = dest_video.read()
        i += 1


BlendFunc = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def process_frame(frame: np.ndarray, mos: MosaicUnfair, blend_func: BlendFunc, blending_level: float, file=None):
    frame = frame * np.float32(1/255.0)
    collage = mos.process_dest_img(frame, file=file)[0]
    collage = blend_func(collage, frame, 1.0 - blending_level)
    collage *= 255.0
    return collage.astype(np.uint8)


def frame_process(mos: MosaicUnfair, blend_func: BlendFunc, blending_level: float, in_q: mp.Queue, out_q: mp.Queue):
    while True:
        i, frame = in_q.get()
        if i is None:
            break
        out_q.put((i, process_frame(frame, mos, blend_func, blending_level)))


def enable_gpu(show_warning=True):
    global cupy_available, cp, fast_sq_euclidean, fast_cityblock, fast_chebyshev
    try:
        import cupy as cp
        cupy_available = True

        @cp.fuse
        def fast_sq_euclidean(Asq, Bsq, AB):
            return Asq + Bsq - 2*AB

        fast_cityblock = cp.ReductionKernel(
            'T x, T y',  # input params
            'T z',  # output params
            'abs(x - y)',  # map
            'a + b',  # reduce
            'z = a',  # post-reduction map
            '0',  # identity value
            'fast_cityblock'  # kernel name
        )

        fast_chebyshev = cp.ReductionKernel(
            'T x, T y',  # input params
            'T z',  # output params
            'abs(x - y)',  # map
            'max(a, b)',  # reduce
            'z = a',  # post-reduction map
            '0',  # identity value
            'fast_chebyshev'  # kernel name
        )

    except ImportError:
        if show_warning:
            print("Warning: GPU acceleration enabled with --gpu but cupy cannot be imported. Make sure that you have cupy properly installed. ")


def check_dup_valid(dup):
    assert dup > 0, "dup must be a positive integer or a real number between 0 and 1"
    return dup


def main(args):
    global LIMIT
    num_process = max(1, args.num_process)
    if args.video and not args.gpu:
        LIMIT = (args.mem_limit // num_process) * 2**20
    else:
        LIMIT = args.mem_limit * 2**20

    if len(args.out) > 0:
        folder, file_name = os.path.split(args.out)
        if len(folder) > 0:
            assert os.path.isdir(folder), "The output path {} does not exist!".format(folder)
        # ext = os.path.splitext(file_name)[-1]
        # assert ext.lower() == ".jpg" or ext.lower() == ".png", "The file extension must be .jpg or .png"
    if args.quiet:
        sys.stdout = open(os.devnull, "w")
    
    dup = check_dup_valid(args.dup)

    with mp.Pool(max(1, num_process)) as pool:
        imgs = read_images(args.path, args.size, args.recursive, pool, args.resize_opt, args.auto_rotate)
        
        if len(args.dest_img) == 0: # sort mode
            if args.exp:
                sort_exp(pool, args, imgs)
            else:
                collage, tile_info = make_collage(*sort_collage(imgs, args.ratio, args.sort, args.rev_sort), args.rev_row)
                save_img(collage, args.out, "")
                if args.tile_info_out:
                    with open(args.tile_info_out, "w", encoding="utf-8") as f:
                        f.write(tile_info)
            return

    assert os.path.isfile(args.dest_img)
    if args.video:
        assert not (args.salient and not args.unfair), "Sorry, making photomosaic video is unsupported with fair and salient option. "
        assert args.skip_frame >= 1, "skip frame must be at least 1"

        # total_frames = count_frames(args.dest_img, args.skip_frame)
        dest_video = cv2.VideoCapture(args.dest_img)
        ret, frame = dest_video.read()
        assert ret, f"unable to open video {args.dest_img}"
        dest_shape = frame.shape
    else:
        dest_img = imread(args.dest_img)
        dest_shape = dest_img.shape

    if args.gpu:
        enable_gpu()

    if args.exp:
        assert not args.salient
        assert args.unfair
        unfair_exp(dest_img, args, imgs)        
        return
    
    if args.salient:
        if args.unfair:
            mos = MosaicUnfair(
                dest_shape, imgs, args.max_width, args.colorspace, args.metric,
                args.lower_thresh, args.background, args.freq_mul, not args.deterministic)
        else:
            mos = MosaicFairSalient(dest_shape, imgs, dup, args.colorspace, args.metric, args.lower_thresh, args.background)
    else:
        if args.unfair:
            mos = MosaicUnfair(
                dest_shape, imgs, args.max_width, args.colorspace, args.metric, 
                None, None, args.freq_mul, not args.deterministic)
        else:
            mos = MosaicFair(dest_shape, imgs, dup, args.colorspace, args.metric)
    
    if args.blending == "alpha":
        blend_func = alpha_blend
    else:
        blend_func = brightness_blend

    if args.video:
        th, tw, _ = mos.imgs[0].shape
        res = (tw * mos.grid[0], th * mos.grid[1])
        print("Photomosaic video resolution:", res)
        video_writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), dest_video.get(cv2.CAP_PROP_FPS) / args.skip_frame, res)
        frames_gen = frame_generator(ret, frame, dest_video, args.skip_frame)
        if args.gpu:
            with open(os.devnull, "w") as null:
                for frame in tqdm(frames_gen, desc="[Computing frames]", unit="frame"):
                    video_writer.write(process_frame(frame, mos, blend_func, args.blending_level, null))
        else:
            in_q = mp.Queue(1)
            out_q = mp.Queue()
            processes = []
            for i in range(num_process):
                p = mp.Process(target=frame_process, args=(mos, blend_func, args.blending_level, in_q, out_q)) 
                p.start()
                processes.append(p)

            last_frame = 0
            buffer = dict()
            pbar = tqdm(desc="[Computing frames]", unit="frame")

            def check_queue():
                nonlocal last_frame
                while not out_q.empty():
                    fid, collage = out_q.get()
                    buffer[fid] = collage
                while last_frame in buffer:
                    collage = buffer[last_frame]
                    del buffer[last_frame]
                    last_frame += 1
                    video_writer.write(collage)
                    pbar.update()

            for i, frame in enumerate(frames_gen):
                in_q.put((i, frame))
                check_queue()
            while last_frame <= i:
                check_queue()
                
            for p in processes:
                in_q.put((None, None))
            for p in processes:
                p.join()
        
        frames_gen.close()
        video_writer.release()
    else:
        collage, tile_info = mos.process_dest_img(dest_img)
        print(tile_info)
        collage = blend_func(collage, dest_img, 1.0 - args.blending_level)
        save_img(collage, args.out, "")
        if args.tile_info_out:
            with open(args.tile_info_out, "w", encoding="utf-8") as f:
                f.write(tile_info)

    pool.close()

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
