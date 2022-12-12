#!/bin/bash
tree -dfi --noreport sfontproj/raw/ | xargs -I{} mkdir -p "./raw/{}"
find sfontproj/raw/ -iname "*.png" | xargs -i{} convert {} -background white -alpha remove -alpha off -depth 8 -type Grayscale -threshold 90% ./raw/{}.bmp