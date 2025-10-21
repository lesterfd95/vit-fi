docker run --rm --runtime=nvidia --gpus all --shm-size=32gb -it --mount type=bind,src="$(pwd)"/scripts,target=/app vit_fi:1.0 bash
