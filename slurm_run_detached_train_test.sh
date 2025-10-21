#!/bin/bash
#SBATCH --partition=P2
#SBATCH --nodes=1                 # Número de nodos
#SBATCH --nodelist=c03
             
#SBATCH --ntasks=1                # Número de tareas
#SBATCH --mem=32G                 # Memoria solicitada
#SBATCH --time=72:00:00           # Tiempo máximo de ejecución (hh:mm:ss)

#SBATCH --job-name=mi_docker_job  # Nombre del job
#SBATCH --output=salida_%j.log    # Archivo de salida
#SBATCH --error=error_%j.log      # Archivo de error

#nvidia-smi
#docker load -i ../vit_fi.tar
#docker images
#docker run --rm --runtime=nvidia --gpus all --shm-size=32gb --mount type=bind,src="$(pwd)",target=/app vit_fi:1.0 nvidia-smi
docker run --rm --runtime=nvidia --gpus all --shm-size=32gb \
  --mount type=bind,src="$(pwd)"/scripts,target=/app \
  --name vit_fi_job vit_fi:1.0 \
  bash -c "./train_test.sh vit-all-ber-6.5e-8  vit-all-ber-6.5e-8"