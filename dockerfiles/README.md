# Hasktorch Dockerfiles

This directory provides Hasktorch's Dockerfiles and makes ATen packages for both CPU-only and GPU.

## Building docker images

Build a docker image from cpu/gpu.Dockerfile.

### CPU-version

```bash
$ docker build -f ./cpu.Dockerfile -t hasktorch .
```

### CUDA-version

```bash
$ docker build -f ./gpu.Dockerfile -t hasktorch .
```

## Building ATen package for debian/ubuntu

Build a debian/ubuntu package(libaten-dev_0.1.0_amd64.deb) from a Docker image of (deb-cpu/gpu.Dockerfile).

### CPU-version

```bash
$ make cpu-deb
```

### CUDA-version

```bash
$ make gpu-deb
```
