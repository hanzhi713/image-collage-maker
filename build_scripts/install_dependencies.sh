#/bin/bash
if [[ $1 == "macos" || $1 == "linux" ]]; then
  # change this to your anaconda installation path.
    echo "Building on Unix"
    conda create -n collage -y python=3.6 nomkl numpy scipy scikit-learn pillow tqdm
    conda activate collage
    pip install wurlitzer
elif [[ $1 == "windows" ]]; then
    echo "Install dependencies for windows"
    conda create -n collage -y python=3.6 nomkl numpy scipy scikit-learn pillow tqdm -c conda-forge
    conda activate collage
else
    echo "Unsupported platform: " $1
    exit 1
fi
pip install -r requirements.txt
