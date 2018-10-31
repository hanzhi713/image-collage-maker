import os
from os.path import *
import numpy as np
import cv2
import argparse
import random
from tqdm import tqdm
import concurrent.futures as con
import io
import sys


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


class OutputWrapper(io.TextIOWrapper):
    def __init__(self, v=False):
        super().__init__(sys.stdout.buffer, encoding="utf-8")
        self.v = v

    def write(self, s):
        if self.v:
            super().write(s)
            self.flush()

    def flush(self):
        super().flush()


def calculate_grid_size(rw: int, rh: int, num_imgs: int, v: OutputWrapper = OutputWrapper()) -> tuple:
    """
    :param rw: the width of the target image
    :param rh: the height of the target image
    :param num_imgs: number of images available
    :param v: verbose
    :return: an optimal grid size
    """
    possible_wh = []
    # print("Calculating grid size...", file=v)
    for width in range(1, num_imgs):
        height = num_imgs // width
        possible_wh.append((width, height))

    return min(possible_wh, key=lambda x: ((x[0] / x[1]) - (rw / rh)) ** 2)


def make_collage(grid: tuple, sorted_imgs: list or np.ndarray, rev: False,
                 v: OutputWrapper = OutputWrapper()) -> np.ndarray:
    """
    :param grid: grid size
    :param sorted_imgs: list of images sorted in correct position
    :param rev: whether to have opposite alignment for consecutive rows
    :param v: verbose
    :return: a collage
    """
    for i in tqdm(range(grid[1]), desc="[Merging]", unit="row", file=v):
        if rev and i % 2 == 1:
            img_row = sorted_imgs[(i + 1) * grid[0] - 1]
            for j in range(grid[0] - 2, -1, -1):
                img_row = np.append(img_row, sorted_imgs[i * grid[0] + j], axis=1)
            combined_img = np.append(combined_img, img_row, axis=0)
        else:
            img_row = sorted_imgs[i * grid[0]]
            for j in range(1, grid[0]):
                img_row = np.append(img_row, sorted_imgs[i * grid[0] + j], axis=1)
            if i == 0:
                combined_img = img_row
            else:
                combined_img = np.append(combined_img, img_row, axis=0)
    return combined_img


def calculate_decay_weights_normal(shape: tuple, sigma: float = 1) -> np.ndarray:
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
    h_arr = truncnorm.pdf(np.linspace(-1, 1, h), -1, 1, loc=0, scale=abs(sigma))
    w_arr = truncnorm.pdf(np.linspace(-1, 1, w), -1, 1, loc=0, scale=abs(sigma))
    h_arr, w_arr = np.meshgrid(h_arr, w_arr)
    if sigma > 0:
        weights = h_arr * w_arr
    else:
        weights = 1 - h_arr * w_arr
    return weights


def sort_collage(imgs: list, ratio: tuple, sort_method="pca_lab", rev_sort=False,
                 v: OutputWrapper = OutputWrapper()) -> [tuple, np.ndarray]:
    """
    :param imgs: list of images
    :param ratio: The aspect ratio of the collage
    :param sort_method:
    :param rev_sort: whether to reverse the sorted array
    :param v: verbose
    :return: calculated grid size and the sorted image array
    """
    num_imgs = len(imgs)
    result_grid = calculate_grid_size(ratio[0], ratio[1], num_imgs, v)

    print("Calculated grid size based on your aspect ratio:", result_grid, file=v)
    print("Note that", num_imgs - result_grid[0] * result_grid[1], "images will be thrown away from the collage",
          file=v)

    print("Sorting images...", file=v)
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

        # only take the first value if img_keys is a list of tuples
        if type(img_keys[0]) == tuple:
            img_keys = list(map(lambda x: x[0], img_keys))
        img_keys = np.array(img_keys)

    sorted_imgs = np.array(imgs)[np.argsort(img_keys)]
    if rev_sort:
        sorted_imgs = list(reversed(sorted_imgs))

    return result_grid, sorted_imgs


def chl_mean_hsv(weights):
    def f(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.average(img[:, :, 0], weights=weights), \
               np.average(img[:, :, 1], weights=weights), \
               np.average(img[:, :, 2], weights=weights)

    return f


def chl_mean_hsl(weights):
    def f(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return np.average(img[:, :, 0], weights=weights), \
               np.average(img[:, :, 1], weights=weights), \
               np.average(img[:, :, 2], weights=weights)

    return f


def chl_mean_bgr(weights):
    def f(img):
        return np.average(img[:, :, 0], weights=weights), \
               np.average(img[:, :, 1], weights=weights), \
               np.average(img[:, :, 2], weights=weights)

    return f


def chl_mean_lab(weights):
    def f(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return np.average(img[:, :, 0], weights=weights), \
               np.average(img[:, :, 1], weights=weights), \
               np.average(img[:, :, 2], weights=weights)

    return f


def calculate_collage_bipartite(dest_img_path: str, imgs: list, dup: int = 1, colorspace="lab", ctype="float16",
                                sigma: float = 1.0, v: OutputWrapper = OutputWrapper()) -> [tuple, list, float]:
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
    from lap import lapjv

    print("Duplicating {} times".format(dup), file=v)

    # avoid modifying the original array
    imgs = list(map(np.copy, imgs))
    imgs_copy = list(map(np.copy, imgs))
    for i in range(dup - 1):
        imgs.extend(imgs_copy)

    num_imgs = len(imgs)

    # Compute the grid size based on the number images that we have
    dest_img = cv2.imread(dest_img_path)
    rh, rw, _ = dest_img.shape
    result_grid = calculate_grid_size(rw, rh, num_imgs, v)

    print("Calculated grid size based on the aspect ratio of the image provided:", result_grid, file=v)
    print("Note:", num_imgs - result_grid[0] * result_grid[1], "images will be thrown away from the collage", file=v)

    # it's VERY important to remove redundant images
    # this makes sure that cost_matrix is a square
    imgs = imgs[:result_grid[0] * result_grid[1]]

    # Resize the destination image so that it has the same size as the grid
    # This makes sure that each image in the list of images corresponds to a pixel of the destination image
    dest_img = cv2.resize(dest_img, result_grid, cv2.INTER_CUBIC)

    weights = calculate_decay_weights_normal(imgs[0].shape[:2], sigma)

    print("Computing cost matrix...", file=v)
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
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2LAB)
    else:
        raise Exception()

    dest_img = dest_img.reshape(result_grid[0] * result_grid[1], 3)

    # compute pair-wise distances
    cost_matrix = cdist(img_keys, dest_img, metric="euclidean")

    ctype = eval("np." + ctype)
    cost_matrix = ctype(cost_matrix)

    print("Computing optimal assignment on a {}x{} matrix...".format(cost_matrix.shape[0], cost_matrix.shape[1]),
          file=v)

    cost, _, cols = lapjv(cost_matrix)
    print("Total assignment cost:", cost, file=v)

    return result_grid, np.array(imgs)[cols], cost


def calculate_collage_dup(dest_img_path: str, imgs: list, max_width: int = 50, colorspace="lab",
                          sigma: float = 1.0, v: OutputWrapper = OutputWrapper()) -> [tuple, list, float]:
    """
    Compute the optimal assignment between the set of images provided and the set of pixels of the target image,
    given that every image could be used arbitrary amount of times

    :param dest_img_path: path to the dest_img file
    :param imgs: list of images
    :param max_width: max_width of the resulting dest_img
    :param colorspace: colorspace used
    :param sigma:
    :param v: verbose
    :return: [gird size, sorted images, total assignment cost]
    """
    assert isfile(dest_img_path)
    from scipy.spatial.distance import cdist

    dest_img = cv2.imread(dest_img_path)

    # Because we don't have a fixed total amount of images as we can used a single image
    # for arbitrary amount of times, we need user to specify the maximum width in order to determine the grid size.
    rh, rw, _ = dest_img.shape
    rh = round(rh * max_width / rw)
    result_grid = (max_width, rh)

    print("Calculated grid size based on the aspect ratio of the image provided:", result_grid, file=v)

    weights = calculate_decay_weights_normal(imgs[0].shape[:2], sigma)
    dest_img = cv2.resize(dest_img, result_grid, cv2.INTER_CUBIC)

    print("Computing costs", file=v)
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
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2LAB)
    else:
        raise Exception()

    dest_img = dest_img.reshape(result_grid[0] * result_grid[1], 3)

    sorted_imgs = []
    cost = 0
    for pixel in tqdm(dest_img, desc="[Computing assignments]", unit="pixel", unit_divisor=1000, unit_scale=True,
                      file=v):
        # Compute the distance between the current pixel and each image in the set
        dist = cdist(img_keys, np.array([pixel]), metric="euclidean")[:, 0]

        # Find the index of the image which best approximates the current pixel
        idx = np.argmin(dist)

        # Store that image
        sorted_imgs.append(imgs[idx])

        # Accumulate the distance to get the total cot
        cost += dist[idx]

    return result_grid, sorted_imgs, cost


def save_img(img: np.ndarray, path: str, suffix: str, v: OutputWrapper = OutputWrapper()) -> None:
    if len(path) == 0:
        path = 'result_{}.png'.format(suffix)
        print("Saving to", path, file=v)
        cv2.imwrite(path, img)
    else:
        if len(suffix) == 0:
            print("Saving to", path, file=v)
            cv2.imwrite(path, img)
        else:
            path = path.split(".")
            path = path[0] + "_{}".format(suffix) + "." + path[1]
            print("Saving to", path, file=v)
            cv2.imwrite(path, img)


def ss_wrapper(a):
    return sort_collage(*a)


def sc_wrapper(a):
    return calculate_collage_bipartite(*a)


def sc_dup_wrapper(a):
    return calculate_collage_dup(*a)


def read_images(pic_path: str, img_size: tuple, v: OutputWrapper) -> list:
    files = list(os.walk(pic_path))[0][-1]
    try:
        files.remove("cache.pkl")
    except ValueError:
        pass

    imgs = []
    for i, img_file in tqdm(enumerate(files), total=len(files), desc="[Reading images]", unit="imgs", file=v):
        try:
            imgs.append(cv2.resize(cv2.imread(join(pic_path, img_file)), img_size, interpolation=cv2.INTER_CUBIC))
        except:
            imgs.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8))
    return imgs


all_sort_methods = ["none", "bgr_sum", "av_hue", "av_sat", "av_lum", "rand",
                    "pca_bgr", "pca_hsv", "pca_lab", "pca_gray", "pca_lum", "pca_sat", "pca_hue",
                    "tsne_bgr", "tsne_hsv", "tsne_lab", "tsne_gray", "tsne_lum", "tsne_sat", "tsne_hue",
                    "umap_bgr", "umap_hsv", "umap_lab", "umap_gray", "umap_lum", "umap_sat", "umap_hue",
                    ]
all_color_spaces = ["hsv", "hsl", "bgr", "lab"]

all_ctypes = ["float16", "float32", "float64", "int16", "int32"]

if __name__ == "__main__":
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
    parser.add_argument("--colorspace", type=str, default="lab", choices=all_color_spaces,
                        help="Methods to use when fitting an image")
    parser.add_argument("--uneven", action="store_true", help="Whether to use each image only once")
    parser.add_argument("--max_width", type=int, default=50, help="Maximum of the collage")
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
                        help="Type of the cost matrix. "
                             "Float16 is a good compromise between computational time and accuracy",
                        choices=all_ctypes)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--exp", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    v = OutputWrapper(args.verbose)

    imgs = read_images(args.path, (args.size, args.size,), v)

    if len(args.collage) == 0:
        if args.exp:

            pool = con.ProcessPoolExecutor(4)
            futures = []


            def callback(x, out, rev_row, suffix, pbar, v: OutputWrapper = OutputWrapper()):
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

        else:
            result_grid, sorted_imgs = sort_collage(imgs, args.ratio, args.sort, args.rev_sort, v)
            save_img(make_collage(result_grid, sorted_imgs, args.rev_row, v), args.out, "", v)
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
                        f = pool.submit(sc_dup_wrapper, (args.collage, imgs, args.max_width, color_space, sigma, v))
                        futures.append((f, sigma, color_space))
            else:
                for sigma in all_sigmas:
                    for color_space in all_color_spaces:
                        f = pool.submit(sc_wrapper, (args.collage, imgs, args.dup,
                                                     color_space, args.ctype, sigma, v))
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

        else:
            if args.uneven:
                result_grid, sorted_imgs, cost = calculate_collage_dup(args.collage, imgs, args.max_width,
                                                                       args.colorspace, args.sigma, v)
                save_img(make_collage(result_grid, sorted_imgs, args.rev_row, v),
                         args.out, "", v)
            else:
                result_grid, sorted_imgs, cost = calculate_collage_bipartite(args.collage, imgs, args.dup,
                                                                             args.colorspace, args.ctype, args.sigma, v)
                save_img(make_collage(result_grid, sorted_imgs, args.rev_row, v),
                         args.out, "", v)
