#!/usr/bin/env bash

REPOSITORY="argnctu/ros2-gz"
TAG="cpu"

IMG="${REPOSITORY}:${TAG}"

docker pull "${IMG}"
