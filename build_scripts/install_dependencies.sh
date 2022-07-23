#/bin/bash
python3 -m venv collage
PLATFORM=$1
if [[ $PLATFORM == windows* ]]; then
    ./Scripts/activate
elif [[ $PLATFORM == macos* ]]; then
    source collage/bin/activate
elif [[ $PLATFORM == ubuntu* ]]; then
    source collage/bin/activate
else
    echo "Unsupported platform: " $PLATFORM
    exit 1
fi

cat requirements.txt | xargs -n 1 -L 1 pip install