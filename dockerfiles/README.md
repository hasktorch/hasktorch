# Hasktorch Dockerfiles

This directory provides Hasktorch's Dockerfiles and makes ATen packages for both CPU-only and GPU.

## Building docker images

Build a docker image from cpu/gpu.Dockerfile.

### CPU-version

```bash
$ docker build -f ./cpu.Dockerfile -t hasktorch:cpu .
```

### CUDA-version

```bash
$ docker build -f ./gpu.Dockerfile -t hasktorch:gpu .
```

## Building ATen package for debian/ubuntu

Build a debian/ubuntu package(libaten-dev_0.1.0_amd64.deb) from a Docker image of (cpu/gpu.Dockerfile).
```debuild```-command uses ```cpu/gpu.Makefile``` and ```debian```-directory to make a debian/ubuntu package.
```cpu/gpu.Makefile``` just copies files of ```/hasktorch/ffi/deps/aten/build``` to the right place by using $(DESTDIR) of a environment variable.

### CPU-version

```bash
$ docker run --name tmp hasktorch:cpu bash -c 'cd /tmp/app;debuild -ePATH -uc -us -i -b'
$ docker cp tmp:/tmp/libaten-dev_0.1.0_amd64.deb .
$ docker rm tmp
```

### CUDA-version

```bash
$ docker run --name tmp hasktorch:gpu bash -c 'cd /tmp/app;debuild -ePATH -uc -us -i -b'
$ docker cp tmp:/tmp/libaten-dev_0.1.0_amd64.deb .
$ docker rm tmp
```
