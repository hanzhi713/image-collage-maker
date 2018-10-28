#!/usr/bin/env bash
python make_img.py --path img --out collage-best-fit.png --collage img/1.png --size 20 --uneven --max_width 62 --verbose
python make_img.py --path img --out collage.png --collage img/1.png --size 20 --dup 10 --verbose