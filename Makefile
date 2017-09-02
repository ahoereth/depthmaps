WD = /home/ubuntu
IMAGE = tensorflow/tensorflow:1.3.0-gpu-py3
MACHINE ?= tfcc

all: rsync
	nvidia-docker run -it -v $(WD):/wd -w /wd $(IMAGE) python3 src/run.py

tensorboard:
	docker run -d -v $(WD)/logs:/logs -w /logs -p 6006:6006 \
		tensorflow/tensorflow:nightly tensorboard --logdir /logs --port 6006

gpustat:
	nvidia-docker run -it nvidia/cuda watch -n 1 nvidia-smi

rsync:
	rsync -av --exclude=.git --filter=":- .gitignore" -e "docker-machine ssh $(MACHINE)" . :
