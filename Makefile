output/%/video.mp4: \
	output/%/000.png
	cd output/$*/
	ffmpeg -framerate 20 -pattern_type glob -i '*.png' -c:v libx264 -qp 16 -pix_fmt gray video.mp4

# <meteor>/<observer
output/%/000.png: \
	render.py \
	default.yaml \
	output/%.yaml
	$(eval words := $(subst /, ,$*))
	$(eval meteor := $(word 1,$(words)))
	$(eval observer := $(word 2,$(words)))
	mkdir -p $(dir $@)
	./render.py default.yaml output/$(meteor)/$(observer).yaml config/observers/$(observer).yaml config/renderers/default.yaml output/$(meteor)/$(observer)/ -p config/projections/zero.yaml -j 20

# <meteor>
output/%/meteor.yaml: \
	simulate.py
	mkdir -p $(dir $@)
	./simulate.py default.yaml config/meteors/$*.yaml -o $@

# <meteor>/<observer>
output/%.yaml: \
	observe.py
	mkdir -p $(dir $@)
	$(eval words := $(subst /, ,$*))
	$(eval meteor := $(word 1,$(words)))
	$(eval observer := $(word 2,$(words)))
	mkdir -p $(dir $@)
	./observe.py default.yaml output/$(meteor).yaml config/observers/$(observer).yaml -o $@
