#!/bin/sh
VERSION=2.1
NAME="collage-maker-macos-x64"
rm -rf macos_builds/$VERSION
pyinstaller -y --onefile --additional-hooks-dir . --distpath macos_builds/$VERSION \
--exclude-module umap --name "$NAME" ../gui.py
pyinstaller -y --additional-hooks-dir . --distpath macos_builds/$VERSION/archive \
--exclude-module umap --name "$NAME"  ../gui.py
tar -czvf "macos_builds/$VERSION/$NAME.tar.gz" -C "macos_builds/$VERSION/archive" "$NAME"