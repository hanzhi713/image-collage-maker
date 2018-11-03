#!/usr/bin/env bash
python make_img.py --path img/zhou --out collage.png --collage img/zhou/1.jpg --size 20 --verbose --dup 10
python make_img.py --path img/zhou --out collage-best-fit.png --collage img/zhou/1.jpg --size 20 --uneven --verbose --max_width 62