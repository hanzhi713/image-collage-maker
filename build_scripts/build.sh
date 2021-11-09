#!/bin/bash
VERSION=3.0
PLATFORM=$1
NAME=photomosaic-maker-${VERSION}-$PLATFORM-x64

if [[ -z "${CONDA}" ]]; then
    # change this to your anaconda installation path.
    CONDA=$HOME/anaconda3
fi

echo "Using conda path: $CONDA"
rm -rf dist/*

if [[ $PLATFORM == "windows" ]]; then
    SUFFIX=".exe"
elif [[ $PLATFORM == "macos" ]]; then
    SUFFIX=""
elif [[ $PLATFORM == "ubuntu" ]]; then
    SUFFIX=""
else
    echo "Unsupported platform: " $PLATFORM
    exit 1
fi

conda activate collage
pyinstaller --hidden-import='PIL._tkinter_finder' -y $ADD_ARGS --exclude-module umap --name "${NAME}-archive" gui.py
pyinstaller --hidden-import='PIL._tkinter_finder' -y --onefile $ADD_ARGS --exclude-module umap --name "$NAME${SUFFIX}" gui.py
tar -czvf "dist/$NAME-archive.tar.gz" -C dist $NAME-archive
