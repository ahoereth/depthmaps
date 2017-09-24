WD ?= $(shell pwd)
IMAGE = tensorflow/tensorflow:nightly-gpu-py3
MACHINE ?= tfcc
STATE ?= d

run:
	nvidia-docker run -${STATE} -v $(WD):/wd -w /wd $(IMAGE) python run.py ${ARGS}

# Visualize learning progress -- stdout is mostly mute, use this instead.
tensorboard:
	docker run -${STATE} -v $(WD)/logs:/logs -w /logs -p 6006:6006 \
		tensorflow/tensorflow:nightly tensorboard --logdir /logs --port 6006

# Check remote GPU usage.
gpustat:
	nvidia-docker run -it nvidia/cuda watch -n 1 nvidia-smi

rsync:
	rsync -av --exclude=.git --filter=":- .gitignore" -e "docker-machine ssh $(MACHINE)" . :

# Quick access for a visual insight into the datasets.
datasets:
	python3 -m datasets.merged
datasets/%:
	python3 -m datasets.$*
