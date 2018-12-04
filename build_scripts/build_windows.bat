@echo off
set NAME="collage-maker-windows-x64"
set VERSION=2.0
rmdir windows_builds\%VERSION% /s /q
pyinstaller -y --additional-hooks-dir . --distpath windows_builds/%VERSION%/archive --exclude-module umap --name %NAME% ../gui.py
pyinstaller -y --onefile --additional-hooks-dir . --distpath windows_builds/%VERSION% --exclude-module umap --name %NAME% ../gui.py