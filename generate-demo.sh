#!/bin/sh
python make_img.py --sort rand --size 50
python make_img.py --sort av_hue --size 50
python make_img.py --sort av_sat --size 50
python make_img.py --sort bgr_sum --size 50
python make_img.py --sort lab --size 50
python make_img.py --sort bgr --size 50
python make_img.py --sort lum --size 50
python make_img.py --sort pca_bgr --size 50
python make_img.py --sort pca_hsv --size 50
python make_img.py --sort pca_lab --size 50