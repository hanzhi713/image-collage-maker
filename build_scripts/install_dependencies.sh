#/bin/bash
if [[ $1 == "macos" || $1 == "ubuntu" ]]; then
  # change this to your anaconda installation path.
    echo "Building on Unix"
    conda activate collage
    pip install wurlitzer
elif [[ $1 == "windows" ]]; then
    echo "Install dependencies for windows"
    conda activate collage
else
    echo "Unsupported platform: " $1
    exit 1
fi
pip install -r requirements.txt
