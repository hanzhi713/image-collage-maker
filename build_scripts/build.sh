#!/bin/bash
VERSION=3.0
PLATFORM=$1
NAME=photomosaic-maker-${VERSION}-$PLATFORM-x64

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

ARGS="--hidden-import='PIL._tkinter_finder' -y --exclude-module umap --exclude-module matplotlib"
pyinstaller $ARGS --name "${NAME}-archive" gui.py
pyinstaller $ARGS --onefile --name "$NAME${SUFFIX}" gui.py

pushd dist/$NAME-archive
tar -czvf ../$NAME.tar.gz .
# zip -r ../$NAME.zip .
popd