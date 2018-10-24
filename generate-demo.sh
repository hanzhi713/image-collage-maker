#!/bin/sh
python make_img.py --sort rand --size 50
python make_img.py --sort av_hue --size 50
python make_img.py --sort av_sat --size 50
python make_img.py --sort bgr_sum --size 50
python make_img.py --sort av_lum --size 50

python make_img.py --sort pca_bgr --size 50
python make_img.py --sort pca_hsv --size 50
python make_img.py --sort pca_lab --size 50
python make_img.py --sort pca_gray --size 50
python make_img.py --sort pca_lum --size 50
python make_img.py --sort pca_sat --size 50
python make_img.py --sort pca_hue --size 50

python make_img.py --sort tsne_bgr --size 50
python make_img.py --sort tsne_hsv --size 50
python make_img.py --sort tsne_lab --size 50
python make_img.py --sort tsne_gray --size 50
python make_img.py --sort tsne_lum --size 50
python make_img.py --sort tsne_sat --size 50
python make_img.py --sort tsne_hue --size 50

python make_img.py --sort umap_bgr --size 50
python make_img.py --sort umap_hsv --size 50
python make_img.py --sort umap_lab --size 50
python make_img.py --sort umap_gray --size 50
python make_img.py --sort umap_lum --size 50
python make_img.py --sort umap_sat --size 50
python make_img.py --sort umap_hue --size 50