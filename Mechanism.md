# Mechanism

## Sorting

Image sorting is based on either the average/sum of single/multiple channels of the color space used or dimensionality reduction methods.

### Channel sum or average

```python
# Simply compute the sum of all RGB values for every source image and sort using the sum as the key.
def bgr_sum(img: np.ndarray) -> float:
    return np.sum(img)


# Or simply compute the average of a particular channel of a particular color space
# In this case it's the average of hues
def av_hue(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0])

# maps one of such functions to the list of images
img_keys = list(map(bgr_sum, imgs))

# only take the first value if img_keys is a list of tuples
if type(img_keys[0]) == tuple:
    img_keys = list(map(lambda x: x[0], img_keys))
img_keys = np.array(img_keys)

sorted_imgs = np.array(imgs)[np.argsort(img_keys)]
```

### Dimensionality reduction

The dimensionality reduction techniques available to use include [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis), [t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://lvdmaaten.github.io/tsne/) and [Uniform Manifold Approximation and Projection](https://github.com/lmcinnes/umap). They will be applied to an array of flattened images (reshape from (n, h, w, 3) to (n, w\*h)), producing a 1-D array of size n.

```python
# sort_function converts each image of shape (h, w, 3)
# to a flattened array of shape (1, h * w)
flattened_imgs = np.array(list(map(sort_function, imgs)))

# Use one of the selected dimensionality reduction techniques
# to reduce each image to a simple number
img_keys = PCA(1).fit_transform(flattened_imgs)[:, 0]

# get the final sorted array based on the sorting result of the img_keys
sorted_imgs = np.array(imgs)[np.argsort(img_keys)]
```

## Image Fitting

### Even distribution

Without the `--uneven` option, a collage will be produced with each image being used for the same number of times. This number depends on your `--dup` option.

The collage maker, in this case, will manage to find an optimal assignment between the source images and the pixels of the destination image. It's based on the minimal weight bipartite matching (linear sum assignment) on a cost matrix of color distances. I used [an implementation of the Jonker-Volgenant algorithm](https://github.com/src-d/lapjv) to solve this problem. It runs in O(n<sup>3</sup>) and can solve 5000x5000 case in less than 30s using float32.

```python
# Compute the grid size based on the number images that we have
dest_img = cv2.imread(dest_img_path)
rh, rw, _ = dest_img.shape
result_grid = calculate_grid_size(rw, rh, num_imgs, v)

# Resize the destination image so that it has the same size as the grid
# This makes sure that each source image corresponds to a pixel of the destination image
dest_img = cv2.resize(dest_img, result_grid, cv2.INTER_AREA)

"""
Each pixel of the destination image is represented as a 3D vector [a, b, c]
where the numerical values of a, b and c depend on the color space used.
We need to map each image, which is an array of shape (h, w, 3),
to also 3-D vector [a, b, c], so we can compute the distance between them.
"""

# Currently, this mapping function is just a channel-wise weighted average of the color space used
def chl_mean_lab(weights):
    def f(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return np.average(img[:, :, 0], weights=weights), \
               np.average(img[:, :, 1], weights=weights), \
               np.average(img[:, :, 2], weights=weights)

    return f

img_keys = np.array(list(map(chl_mean_lab(weights), imgs)))

# have to make sure that the dest_img also uses the same color space representation
dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2LAB)

# reshape dest_img to an 1-D array of pixels to match the shape of the img_keys
dest_img = dest_img.reshape(result_grid[0] * result_grid[1], 3)

# compute the pair-wise distances to construct the cost matrix
# using the Euclidean distance metric
cost_matrix = cdist(img_keys, dest_img, metric="euclidean")

# compute the minimum weight bipartite matching to get the optimal assignment
cost, _, cols = lapjv(cost_matrix)

# get the sorted array from the column indices of the optimal assignment
sorted_imgs = np.array(imgs)[cols]
```

### Uneven Distribution

An image can be better fitted if we don't restrict the number of times that each image should be used.

```python
dest_img = cv2.imread(dest_img_path)

# Because we don't have a fixed total amount of images as we can use a single image for
# arbitrary amount of times, we need the user to specify the maximum width in order to determine the grid size.
rh, rw, _ = dest_img.shape
rh = round(rh * max_width / rw)
result_grid = (max_width, rh)

# similar preparation steps
dest_img = cv2.resize(dest_img, result_grid, cv2.INTER_AREA)
def chl_mean_lab(weights):
    def f(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return np.average(img[:, :, 0], weights=weights), \
               np.average(img[:, :, 1], weights=weights), \
               np.average(img[:, :, 2], weights=weights)

    return f
img_keys = np.array(list(map(chl_mean_lab(weights), imgs)))
dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2LAB)
dest_img = dest_img.reshape(result_grid[0] * result_grid[1], 3)

sorted_imgs = []
cost = 0
for pixel in dest_img:
    # Compute the distance between the current pixel and each image in the set
    dist = cdist(img_keys, np.array([pixel]), metric="euclidean")[:, 0]

    # Find the index of the image which best approximates the current pixel
    idx = np.argmin(dist)

    # Store that image
    sorted_imgs.append(imgs[idx])

    # Accumulate the distance to get the total cost
    cost += dist[idx]

```

### Salient Object Only

Displaying only salient object in an image utilizes the saliency object in the OpenCV library.

#### Salient object only for even distribution

First, we compute a binary image `thresh` to indicate the location of the pixel in the destination image that constitutes a salient object.

```python

# create a saliency object.
saliency = cv2.saliency.StaticSaliencyFineGrained_create()

"""
generate an image that depicts the saliency of objects
in an image with a float number from 0 to 1; the larger
the number, the more salient the corresponding pixel.
"""
_, saliency_map = saliency.computeSaliency(dest_img)

"""
generate a binary image. the pixel is white, i.e. has a value of 255,
if the corresponding pixel in saliency_map is greater than the threshold.
Otherwise the pixel is black, i.e. has a value of 0.
"""
_, thresh = cv2.threshold(saliency_map * 255, lower_thresh, 255, cv2.THRESH_BINARY)

# store the number of pixels that constitute an object
obj_area = np.count_nonzero(thresh.astype(np.uint8))
```

Since we want to use as many source images as possible, and the number of pixels that constitutes an object (i.e. object area) cannot exceed the number of source images, we have to recalculate the size of the destination image according to the number of source images and the object area.

However, the ratio of the object area to the total area may be different after resizing. Thus, we use a while loop to adjust the threshold dynamically, in order to make the number of source images and the object area close enough. Once they are convergent, and the object area does not exceed the number of source images, we no longer have to resize the destination image or change the threshold, and the resized destination image is ready to be filtered to depict salient object only.

```python
while True:
        """
        calculate the total number of image based on the number
        of source images and number of pixels that
        constitutes an object.
        """
        num_imgs = round(rh * rw / obj_area * len(imgs))

        grid = calc_grid_size(rw, rh, num_imgs)

        dest_img = cv2.resize(dest_img_copy, grid, cv2.INTER_AREA)

        """
        again, generate the binary graph and calculate
        object area after resized.
        """
        saliency2 = cv2.saliency.StaticSaliencyFineGrained_create()
        _, saliency_map_resized = saliency2.computeSaliency(dest_img)
        _, thresh_resized = cv2.threshold(
            saliency_map_resized * 255, threshold, 255, cv2.THRESH_BINARY)

        rh, rw, _ = dest_img.shape

        thresh_resized = thresh_resized.astype(np.uint8)
        obj_area = np.count_nonzero(thresh_resized)


        diff = len(imgs) - obj_area

        pbar.update(1)

        """
        update threshold based on the difference
        between the number of source images and the object area.
        if object area is smaller than the number of
        images, we have to use a lower threshold so
        that more pixels would be detected as a component
        of objects, vice versa.
        """
        if threshold != -1:
            if diff > 0 :
                threshold -= 2
                if threshold < 1:
                    threshold = 1
            else:
                threshold += 2
                if threshold > 254:
                    threshold = 254

        """
        if the difference is small enough, and the
        number of pixels is less than the number of
        images, we no longer have to adjust the size of
        the destination image and recalculate the object area.
        """
        if diff >= 0 and diff < int(len(imgs) / dup / 2) or pbar.n > 100:
            break
```

Record the coordinate and color value of the pixels that constitutes an object.

```python
dest_obj = []
coor = []

for i in range(rh):
    for j in range(rw):
        if thresh_resized[i, j] != 0:
            coor.append(i * rw + j)
```

Compute the optimal assignment and make the collage.

```python
_, cols, cost = lapjv(cost_matrix)

paired = np.array(imgs)[cols]

white = np.ones(imgs[0].shape, np.uint8)
white[:, :, :] = background

filled = []
counter = 0
for i in range(grid[0] * grid[1]):

    """
    if the pixel indicated by i is one that constitutes an object, append its
    corresponding source image to the collage
    """
    if i in coor:
        filled.append(paired[counter])
        counter += 1
    else:
        filled.append(white)
```

#### Salient object only for uneven distribution

There are only two notable differences between this option and the uneven option without the `--salient` flag.

Convert the destination image into a binary image, and extract area containing salient objects using the binary image.

```python
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency.computeSaliency(dest_img)
    _, thresh = cv2.threshold(
        saliency_map * 255, lower_thresh, 255, cv2.THRESH_BINARY)

```

```python
for i in range(rh):
        for j in range(rw):
            if thresh[i, j] < 10:
                # background is a tuple of RGB values
                dest_img[i, j, :] = np.array(background, np.uint8)
```

Add a blank image with the designated background color to the array of source images.

```python
white = np.ones(imgs[0].shape, np.uint8)
white[:, :, :] = background
imgs.append(white)
```
