docker run -d --rm --runtime=nvidia --gpus all --shm-size=32gb \
  --mount type=bind,src="$(pwd)"/scripts,target=/app \
  --name vit_fi_job vit_fi:1.0 \
  bash -c "./test.sh test-vit-cifar10-16-ckpt-multilayer vit-cifar10-16-ckpt"