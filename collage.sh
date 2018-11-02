#!/usr/bin/env bash
python make_img.py --path img/zhou --out collage-best-fit.png --collage img/zhou/1.jpg \
--size 20 --uneven --max_width 62 --verbose

python make_img.py --path img/zhou --out collage.png --collage img/zhou/1.jpg --size 20 --dup 10 --verbose