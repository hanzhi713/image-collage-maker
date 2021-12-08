#!/bin/bash
CMD="echo python make_img.py --path img/zhou"

$CMD --sort none --size 50 --out examples/unsorted.png
$CMD --sort bgr_sum --size 50 --out examples/sort-bgr.png
$CMD --dest_img examples/dest.jpg --size 25 --dup 8 --out examples/fair-dup-10.png
$CMD --dest_img examples/dest.jpg --size 25 --unfair --max_width 56 --out examples/best-fit.png

$CMD --dest_img examples/messi.jpg --size 25 --salient --lower_thresh 0.15 --dup 5 --out examples/messi-fair.png
$CMD --dest_img examples/messi.jpg --size 25 --salient --lower_thresh 0.15 --unfair --max_width 115 --out examples/messi-unfair.png
$CMD --dest_img examples/dest.jpg --size 25 --dup 8 --blending alpha --blending_level 0.25 --out examples/blend-alpha-0.25.png
$CMD --dest_img examples/dest.jpg --size 25 --dup 8 --blending brightness --blending_level 0.25 --out examples/blend-brightness-0.25.png

$CMD --dest_img examples/dest.jpg --size 25 --exp --unfair --max_width 56