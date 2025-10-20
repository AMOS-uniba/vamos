output/%/video.mp4: \
	output/%/000.png
	cd output/$*/
	ffmpeg -framerate 20 -pattern_type glob -i '*.png' -c:v libx264 -qp 16 -pix_fmt gray video.mp4

output/%/000.png: \
	render.py \
	default.yaml \
	output/%.yaml
	./render.py default.yaml output/%.yaml configs/renderers/default.yaml output/%/ -c 20

output/%/meteor.yaml:
	mkdir -p $(dir $@)
	./simulate.py default.yaml -o $@
