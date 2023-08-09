WORK_DIR=${PWD}
PROJECT=pf_track
DOCKER_IMAGE=bcheong/${PROJECT}:latest
DOCKER_FILE=docker/Dockerfile-pftrack
DATA_ROOT_LOCAL=/media/brian/Data2/nuscenes/v1.0-mini
DATA_ROOT_APOLLO=/scratch/hpc_nas/datasets/nuscenes/v1.0-trainval
OUTPUT_APOLLO=/home/bcheong/job_artifacts
CKPTS_ROOT=${PWD}/ckpts

DOCKER_OPTS = \
	-it \
	--rm \
	-e DISPLAY=${DISPLAY} \
	-v /tmp:/tmp \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /mnt/fsx:/mnt/fsx \
	-v ~/.ssh:/root/.ssh \
	-v ~/.aws:/root/.aws \
	-v ${WORK_DIR}:/workspace/${PROJECT} \
	-v ${CKPTS_ROOT}:/workspace/${PROJECT}/ckpts \
	--shm-size=1G \
	--ipc=host \
	--network=host \
	--pid=host \
	--privileged

DOCKER_BUILD_ARGS = \
	--build-arg AWS_ACCESS_KEY_ID \
	--build-arg AWS_SECRET_ACCESS_KEY \
	--build-arg AWS_DEFAULT_REGION \
	--build-arg WANDB_ENTITY \
	--build-arg WANDB_API_KEY \

docker-build:
	nvidia-docker image build \
	-f $(DOCKER_FILE) \
	-t $(DOCKER_IMAGE) \
	$(DOCKER_BUILD_ARGS) .

docker-dev-local:
	docker run \
	--runtime=nvidia \
	--gpus all \
	--name $(PROJECT) \
	-v ${DATA_ROOT_LOCAL}:/workspace/${PROJECT}/data/nuscenes \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

docker-dev-apollo:
	nvidia-docker run --name $(PROJECT) \
	-v ${DATA_ROOT_APOLLO}:/workspace/${PROJECT}/data/nuscenes \
	-v ${OUTPUT_APOLLO}:/workspace/${PROJECT}/work_dirs \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

clean:
	find . -name '"*.pyc' | xargs sudo rm -f && \
	find . -name '__pycache__' | xargs sudo rm -rf