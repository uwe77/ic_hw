#!/usr/bin/env bash

REPOSITORY="argnctu/ros2-gz"
TAG="gpu"

IMG="${REPOSITORY}:${TAG}"

docker pull "${IMG}"
