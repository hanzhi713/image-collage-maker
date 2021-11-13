<table style="border: 0; text-align: center">
    <tr>
        <td>Tiles</td>
        <td>Photomosaic (Fair tile usage)</td>
    </tr>
    <tr>
        <td><img src="examples/result-rand.png" width="480px"></td>
        <td><img src="examples/dest-fair-48x49.png" width="270px"></td>
    </tr>
    <tr>
        <td>Tiles Sorted by RGB sum</td>
        <td>Photomosaic (Best-fit)</td>
    </tr>
    <tr>
        <td><img src="examples/result-pca_bgr.png" width="480px"></td>
        <td><img src="examples/dest-best-fit-49x49.png" width="270px"></td>
    </tr>
</table>

# Photomosaic Maker

<img src="./examples/gui.png" width="720px" alt="gui demo">

## Getting Started

You can either use our pre-built binary files or directly run our python script.

### Using the pre-built binary

Binaries can be downloaded from [release](https://github.com/hanzhi713/image-collage-maker/releases).

On Windows, my program may be blocked by Windows Defender because it is not signed. Don't worry as there is no security risk. On MacOS or Linux, after downloading the binary, you may need to add executing permission. Go to the file's directory and type

```bash
chmod +x ./photomosaic-maker-macos-x64
```

Then you can run from terminal as

```bash
./photomosaic-maker-macos-x64
```

### Running Python script directly

First, you need Python >= 3.6 with pip. You can install dependencies by running

```bash
pip3 install -r requirements.txt
```

Then, you can launch the GUI by running

```bash
python3 gui.py
```

## Command line usage

> If you do not wish to use the GUI, a command line interface is also available. 

<!-- 
>Note: If you already have a set of images to work with, you can skip step 1. The collage maker can be applied to any folder which contains a sufficient amount of images, not limited to your WeChat friends' profile pictures.

### 1\. Use [extract_img.py](extract_img.py) to download profile pictures of your WeChat friends

Download all your friends' profile pictures (--dir specifies the directory to store these images):

```bash
python3 extract_img.py --dir img
```

Or, download the group members' images in a group chat (replace ```name``` with the group chat's name and keep the double quotes):

```bash
python3 extract_img.py --dir img2 --type chatroom --name "name"
```

Sometimes the download may fail, especially when the program is running for the first time. In such case, you need to rerun program with an additional ```--clean``` flag

```bash
python3 extract_img.py --dir img --clean
``` -->

### Option 1: Sorting

```bash
python3 make_img.py --path img --sort pca_bgr --size 50
```

Use ```--ratio w h``` to change the aspect ratio, whose default is 16:9

Example: use ```--ratio 21 9``` to change the aspect ratio to 21:9

Result:

<img src="examples/result-pca_bgr.png"/>

### Option 2: Make a photomosaic

To make a photomosaic, specify the path to the destination image using `--dest_img`

#### Option 2.1: Give a fair chance to each image

This fitting option ensures that each image is used for the same amount of times.

```bash
python3 make_img.py --path img --dest_img img/1.png --size 25 --dup 10 --out collage.png
```

```--dup 10``` allows each source image to be used 10 times. Increase that number if you don't have enough source tiles or you want a better fitting result. Note that a large number of tiles may result in long computational time. To make sure the computation completes within a reasonable amount of time, please make sure that you are using less than 6000 tiles after duplication.

| Original                                    | Fitting Result                                 |
| ------------------------------------------- | ---------------------------------------------- |
| <img src="examples/dest.jpg" width="350px"> | <img src="examples\dest-fair-48x49.png" width="350px"> |


#### Option 2.2: Best fit (unfair tile usage)

This fitting option just selects the best subset of tiles you provided to approximate your destination tiles. Each tile in that subset will be used for an arbitrary number of times.

Add `--unfair` flag to enable this option. You can also specify `--max_width` to change the width of the grid. The height will be automatically calculated based on the max_width provided. Generally, a larger grid will give a better result. The default value is 80.

```bash
python3 make_img.py --path img --out best-fit.png --dest_img img/1.png --size 25 --unfair
```


| Original                                    | Fitting Result                                                   |
| :-------------------------------------------: | :--------------------------------------------------------------: |
| <img src="examples/dest.jpg" width="350px"> | <img src="examples/dest-best-fit-49x49.png" width="350px"> |

Optionally, you can specify the `--freq_mul` parameter that trade-off between the fairness of the tiles and quality of the mosaic. The larger the `freq_mul`, more tiles will be used to construct the photomosaic, but the quality will deteriorate. The results under different `freq_mul` are shown below. Note that if you need a large `freq_mul`, you will better off by going for the fair tile usage (see section above) instead.

<!-- ```bash
python3 make_img.py --path img --out best-fit.png --dest_img img/1.png --size 25 --unfair --freq_mul 1.0
``` -->

![](examples/fairness.png)

#### Option 2.3 Display salient object only

This option makes photomosaic only for the salient part of the destination image. Rest of the area is filled with a background color. 

Add ```--salient``` flag to enable this option. You can still specify whether each image is used for the same amount of times with the ```--unfair``` flag.

Use ```--lower_thresh``` to specify the threshold for object detection. The threshold ranges from 0 to 225; a higher threshold would lead to less object area. The default threshold is 75. If you choose to use each image for the same amount of time, the threshold may have to change so that the number of source images and the number of pixels in the destination can converge. You may use ```--lower_thresh -1``` to enable adaptive thresholding (new in v.2.1).

Use ```--background``` to specify the background color for the collage. The color space for the background option is RGB. The default background color is white, i.e. (255, 255, 255).

```bash
python3 make_img.py --path img --out collage-best-fit.png --dest_img img/1.png --dup 16 --salient
```

```bash
python3 make_img.py --path img --out collage-best-fit.png --dest_img img/1.png --size 25 --salient --unfair
```

| Original                                     | Unfair-Fitting Result                               | Fair-Fitting Result                               |
| -------------------------------------------- | --------------------------------------------------- | ------------------------------------------------- |
| <img src="examples/messi.jpg" width="350px"> | <img src="examples/messi_unfair.png" width="350px"> | <img src="examples/messi_even.png" width="350px"> |

#### Other options

```python3 make_img.py --help``` will give you all the available commandline options.

```
usage: make_img.py [-h] [--path PATH] [--recursive]
                   [--num_process NUM_PROCESS] [--out OUT] [--size SIZE]
                   [--verbose] [--resize_opt {center,stretch}]
                   [--ratio RATIO RATIO]
                   [--sort {none,bgr_sum,av_hue,av_sat,av_lum,rand,pca_bgr,pca_hsv,pca_lab,pca_gray,pca_lum,pca_sat,pca_hue,tsne_bgr,tsne_hsv,tsne_lab,tsne_gray,tsne_lum,tsne_sat,tsne_hue}]
                   [--rev_row] [--rev_sort] [--dest_img DEST_IMG]
                   [--colorspace {hsv,hsl,bgr,lab,luv}]
                   [--metric {euclidean,cityblock,chebyshev}]
                   [--ctypes {float32,float64}] [--unfair]
                   [--max_width MAX_WIDTH] [--redunt_window REDUNT_WINDOW]
                   [--freq_mul FREQ_MUL] [--deterministic] [--dup DUP]
                   [--salient] [--lower_thresh LOWER_THRESH]
                   [--background BACKGROUND BACKGROUND BACKGROUND] [--exp]

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path to the tiles (default: img)
  --recursive           Whether to read the sub-folders for the specified path
                        (default: False)
  --num_process NUM_PROCESS
                        Number of processes to use when loading images
                        (default: 8)
  --out OUT             The filename of the output image (default: )
  --size SIZE           Size (side length) of each tile in pixels (default:
                        50)
  --verbose             Print progress message to console (default: False)
  --resize_opt {center,stretch}
                        How to resize each tile so they become square images.
                        Center: center crop. Stretch: stretch the tile
                        (default: center)
  --ratio RATIO RATIO   Aspect ratio of the output image (default: (16, 9))
  --sort {none,bgr_sum,av_hue,av_sat,av_lum,rand,pca_bgr,pca_hsv,pca_lab,pca_gray,pca_lum,pca_sat,pca_hue,tsne_bgr,tsne_hsv,tsne_lab,tsne_gray,tsne_lum,tsne_sat,tsne_hue}
                        Sort method to use (default: bgr_sum)
  --rev_row             Whether to use the S-shaped alignment. (default:
                        False)
  --rev_sort            Sort in the reverse direction. (default: False)
  --dest_img DEST_IMG   The path to the destination image that you want to
                        build a photomosaic for (default: )
  --colorspace {hsv,hsl,bgr,lab,luv}
                        The colorspace used to calculate the metric (default:
                        lab)
  --metric {euclidean,cityblock,chebyshev}
                        Distance metric used when evaluating the distance
                        between two color vectors (default: euclidean)
  --ctypes {float32,float64}
                        C type of the cost matrix. float32 is a good
                        compromise between computational time and accuracy.
                        Leave as default if unsure. (default: float32)
  --unfair              Whether to allow each tile to be used different amount
                        of times (unfair tile usage). (default: False)
  --max_width MAX_WIDTH
                        Maximum width of the collage. This option is only
                        valid if unfair option is enabled (default: 80)
  --redunt_window REDUNT_WINDOW
                        The guaranteed window size (size x size) to have no
                        duplicated tiles in it (default: 0)
  --freq_mul FREQ_MUL   Frequency multiplier to balance tile fairless and
                        mosaic quality. Minimum: 0. More weight will be put on
                        tile fairness when this number increases. (default: 1)
  --deterministic       Do not randomize the tiles for unfair tile usage
                        (default: False)
  --dup DUP             Duplicate the set of tiles by how many times (default:
                        1)
  --salient             Make photomosaic for salient objects only (default:
                        False)
  --lower_thresh LOWER_THRESH
                        The threshold for saliency detection (default: 127)
  --background BACKGROUND BACKGROUND BACKGROUND
                        Background color in RGB for non salient part of the
                        image (default: (255, 255, 255))
  --exp                 Do experiments (for testing only) (default: False)
```

## Credits (Names in alphabetical order)

Hanzhi Zhou ([hanzhi713](https://github.com/hanzhi713/)): Main algorithm and GUI implementation

Kaiying Shan ([kaiyingshan](https://github.com/kaiyingshan)): Saliency idea and implementation

Xinyue Lin: Idea for the "Best-fit"

Yufeng Chi ([T-K](https://github.com/T-K-233/)) : Initial Idea, crawler
