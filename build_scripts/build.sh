#!/bin/bash
VERSION=3.0
PLATFORM=$1
NAME=photomosaic-maker-${VERSION}-$PLATFORM-x64

if [[ -z "${CONDA}" ]]; then
    # change this to your anaconda installation path.
    CONDA=$HOME/anaconda3
fi

echo "Using conda path: $CONDA"
rm -rf dist

if [[ $PLATFORM == "windows" ]]; then
    SUFFIX=".exe"
elif [[ $PLATFORM == "macos" ]]; then
    SUFFIX=""
elif [[ $PLATFORM == "linux" ]]; then
    ADD_ARGS="--add-data \"$CONDA/envs/collage/lib/python3.6/site-packages/PIL:PIL\""
else
    echo "Unsupported platform: " $PLATFORM
    exit 1
fi

conda activate collage
pyinstaller -y $ADD_ARGS --exclude-module umap --name "${NAME}" gui.py
pyinstaller -y --onefile $ADD_ARGS --exclude-module umap --name "$NAME${SUFFIX}" gui.py
tar -czvf "dist/$NAME.tar.gz" -C dist/$NAME
