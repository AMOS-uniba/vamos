ffmpeg -framerate 20 -pattern_type glob -i '*.png' -c:v libx264 -qp 0 -pix_fmt gray out.mp4
