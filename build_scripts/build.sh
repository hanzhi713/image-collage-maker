#!/bin/bash
PLATFORM=$1
PLATV=$2
if [[ $2 != "" ]]; then
    PLATV="-$2"
fi
NAME=photomosaic-maker-${VERSION}-${PLATFORM}${PLATV}-x64

rm -rf dist/*

if [[ $PLATFORM == "windows" ]]; then
    ./Scripts/activate
    SUFFIX=".exe"
elif [[ $PLATFORM == "macos" ]]; then
    source collage/bin/activate
    SUFFIX=""
elif [[ $PLATFORM == "ubuntu" ]]; then
    source collage/bin/activate
    SUFFIX=""
else
    echo "Unsupported platform: " $PLATFORM
    exit 1
fi

ARGS="-y --exclude-module umap --exclude-module matplotlib"
pyinstaller $ARGS --name "${NAME}-archive" gui.py
pyinstaller $ARGS --onefile --name "$NAME${SUFFIX}" gui.py

pushd dist/$NAME-archive
tar -czvf ../$NAME.tar.gz .
# zip -r ../$NAME.zip .
popd

pushd build
tar -czvf ../$NAME-build.tar.gz .
popd