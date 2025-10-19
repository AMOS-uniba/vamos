cd $1
ffmpeg -framerate 20 -pattern_type glob -i '*.png' -c:v libx264 -qp 16 -pix_fmt gray out.mp4
