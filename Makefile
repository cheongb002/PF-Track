WORK_DIR=${PWD}
PROJECT=pf_track
DOCKER_IMAGE=${PROJECT}:latest
DOCKER_FILE=docker/Dockerfile-pftrack
# apollo params
NUSCENES_ROOT_APOLLO=/scratch/hpc_nas/datasets/nuscenes/v1.0-mini
CKPTS_ROOT_APOLLO=/scratch/hpc_nas/input/bcheong/Projects/PF-Track/ckpts

# local params
NUSCENES_ROOT_LOCAL=/media/brian/Data1
CKPTS_ROOT_LOCAL=./ckpts

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
	nvidia-docker image build -f $(DOCKER_FILE) -t $(DOCKER_IMAGE) \
	$(DOCKER_BUILD_ARGS) .

docker-dev-apollo:
	nvidia-docker run --name $(PROJECT) \
	$(DOCKER_OPTS) \
	-v ${WORK_DIR}:/workspace/${PROJECT} \
	-v ${NUSCENES_ROOT_APOLLO}:/workspace/${PROJECT}/data/nuscenes \
	-v ${CKPTS_ROOT_APOLLO}:/workspace/${PROJECT}/ckpts \
	$(DOCKER_IMAGE) bash

docker-dev-local:
	nvidia-docker run --name $(PROJECT) \
	$(DOCKER_OPTS) \
	-v ${NUSCENES_ROOT_LOCAL}:/workspace/${PROJECT}/data/nuscenes \
	-v ${CKPTS_ROOT_LOCAL}:/workspace/${PROJECT}/ckpts \
	$(DOCKER_IMAGE) bash

clean:
	find . -name '"*.pyc' | xargs sudo rm -f && \
	find . -name '__pycache__' | xargs sudo rm -rf