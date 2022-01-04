IMAGE := autoformer
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	--gpus all \
	-v ${ROOT}:/app \
	-w /app \
	-e HOME=/tmp

init:
	docker build -t ${IMAGE} .

get_dataset:
	mkdir -p dataset/ && \
		make run_module module="python -m utils.download_data" && \
		unzip dataset/datasets.zip -d dataset/ && \
		mv dataset/all_six_datasets/* dataset && \
		rm -r dataset/all_six_datasets dataset/__MACOSX 

run_module: .require-module
	docker run -i --rm ${DOCKER_PARAMETERS} \
		${IMAGE} ${module}

bash_docker:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE}

.require-module:
ifndef module
	$(error module is required)
endif
