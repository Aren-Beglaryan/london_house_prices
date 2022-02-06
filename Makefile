SHELL=/bin/bash
NAME=house-price-prediction
TAG=0.1.0
IMAGE=$(NAME):$(TAG)
MODELPATH=model.pt
ENCODERPATH=encoder
PORT=4562

build:
	@docker build --pull --tag $(IMAGE) .
run:
	@docker run --rm -it -v $(PWD)/$(MODELPATH):/app/service/model.pt -v $(PWD)/$(ENCODERPATH):/app/service/encoder -p $(PORT):8000 --name $(NAME) $(IMAGE)
clean:
	@docker rmi $(IMAGE)
