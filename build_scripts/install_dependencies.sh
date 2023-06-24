#/bin/bash
python3 -m venv collage
PLATFORM=$1
if [[ $PLATFORM == windows* ]]; then
    ./collage/Scripts/activate
elif [[ $PLATFORM == macos* ]]; then
    source collage/bin/activate
elif [[ $PLATFORM == ubuntu* ]]; then
    source collage/bin/activate
else
    echo "Unsupported platform: " $PLATFORM
    exit 1
fi

pip install numpy==1.24.3
cat requirements.txt | xargs -n 1 pip install