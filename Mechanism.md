# Mechanism

## Sorting

Image sorting is based on either the average/sum of single/multiple channels of the color space used or dimensionality reduction methods.

The dimensionality reduction techniques available to use include Principal Component Analysis (PCA), t-Distributed Stochastic Neighbor Embedding (t-SNE) and Uniform Manifold Approximation and Projection (UMAP). They will be applied to an array of flattened images (reshape from (n, h, w, 3) to (n, w\*h)), producing a 1-D array of size n.

```python
# sort_function converts each image of shape (h, w, 3) to a flattened array of shape (1, h * w) 
flattened_imgs = np.array(list(map(sort_function, imgs)))

# Use dimensionality reduction techniques to reduce each image to a simple number
img_keys = PCA(1).fit_transform(flattened_imgs)[:, 0]

# get the final sorted array based on the sorting result of the img_keys
sorted_imgs = np.array(imgs)[np.argsort(img_keys)]
```

## Image Fitting

### Even distribution

Without the ```--uneven``` option, a collage will be produced with each image being used the same amount of times. This number dependents on your ```--dup``` option.

The collage maker in this case will manage to find an optimal pairing between the thumbnail images and pixels of the image that you're trying to fit. It's based on the minimal weight bipartite matching on a cost matrix of color distances.

```python
# Compute the grid size based on the number images that we have
dest_img = cv2.imread(dest_img_path)
rh, rw, _ = dest_img.shape
result_grid = calculate_grid_size(rw, rh, num_imgs, v)

# Resize the destination image so that it has the same size as the grid
# This makes sure that each image in the list of images corresponds to a pixel of the destination image
dest_img = cv2.resize(dest_img, result_grid, cv2.INTER_CUBIC)

"""
Each pixel of the destination image is represented as 3-D vector [a, b, c] 
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

# compute pair-wise distances to construct the cost matrix
cost_matrix = cdist(img_keys, dest_img, metric="euclidean")

# compute the minimum weight bipartite matching to get the optimal assignment
cost, _, cols = lapjv(cost_matrix)
```

### Uneven Distribution

An image can be better fitted if we don't restrict the number of times that each image should be used.

```python
dest_img = cv2.imread(dest_img_path)

# Because we don't have a fixed total amount of images as we can used a single image for 
# arbitrary amount of times, we need user to specify the maximum width in order to determine the grid size.
rh, rw, _ = dest_img.shape
rh = round(rh * max_width / rw)
result_grid = (max_width, rh)

# similar preparation steps
dest_img = cv2.resize(dest_img, result_grid, cv2.INTER_CUBIC)
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
    dist = cdist(img_keys, pixel, metric="euclidean")[:, 0]
    
    # Find the index of the image which best approximates the current pixel
    idx = np.argmin(dist)
    
    # Store that image
    sorted_imgs.append(imgs[idx])
    
    # Accumulate the distance to get the total cot
    cost += dist[idx]

```