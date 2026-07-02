.PHONY: build clean

BUILD_DIR := build
POTTER_ROOT ?= /home/usr1/zhilix/Potter

build:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DPOTTER_ROOT=$(POTTER_ROOT) .
	cmake --build $(BUILD_DIR) -j

clean:
	rm -rf $(BUILD_DIR)
