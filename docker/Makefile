.PHONY: all image

all: image

image:
	docker build -t kamayan/ci:latest -f Dockerfile.ci --platform linux/amd64 .

# docker tag kamayan/ci:latest acreyes/kamayan:v1
# docker tag kamayan/ci:latest acreyes/kamayan:latest
# docker push acreyes/kamayan:v1
# docker push acreyes/kamayan:latest

