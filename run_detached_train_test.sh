docker run -d --rm --runtime=nvidia --gpus all --shm-size=32gb \
  --mount type=bind,src="$(pwd)"/scripts,target=/app \
  --name vit_fi_job vit_fi:1.0 \
  bash -c "./train_test.sh vit-all-ber-6.5e-8  vit-all-ber-6.5e-8"