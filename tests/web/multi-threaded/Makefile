ROOT := ../../../src/cyllama/llama/server
INCLUDES := $(ROOT)


.PHONY: all

all:
	@gcc -o main -I $(INCLUDES) main.c $(ROOT)/mongoose.c
