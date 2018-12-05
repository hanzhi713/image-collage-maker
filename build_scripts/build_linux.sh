#!/bin/sh
VERSION=2.1
NAME="collage-maker-linux-x64"

# change this to your anaconda installation path.
CONDA=$HOME/anaconda3 

rm -rf linux_builds/$VERSION
pyinstaller -y --onefile --additional-hooks-dir . --distpath linux_builds/$VERSION \
--add-data "$CONDA/envs/collage/lib/python3.6/site-packages/PIL:PIL" \
--exclude-module umap --name "$NAME" ../gui.py
pyinstaller -y --additional-hooks-dir . --distpath linux_builds/$VERSION/archive \
--add-data "$CONDA/envs/collage/lib/python3.6/site-packages/PIL:PIL" \
--exclude-module umap --name "$NAME"  ../gui.py
tar -czvf "linux_builds/$VERSION/$NAME.tar.gz" -C "linux_builds/$VERSION/archive" "$NAME"