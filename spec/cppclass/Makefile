all:
	./header2yaml.py --header=../../deps/libtorch/include/c10d/Store.hpp  --namespace=c10d --classname=Store > c10d-store.yaml
	./header2yaml.py --header=../../deps/libtorch/include/c10d/ProcessGroup.hpp  --namespace=c10d --classname=Work > c10d-work.yaml
	./header2yaml.py --header=../../deps/libtorch/include/c10d/ProcessGroup.hpp  --namespace=c10d --classname=ProcessGroup > c10d-processgroup.yaml
	./header2yaml.py --header=../../deps/libtorch/include/c10d/ProcessGroupNCCL.hpp  --namespace=c10d --classname=WorkNCCL > process-group-nccl.yaml
	./header2yaml.py --header=../../deps/libtorch/include/c10d/ProcessGroupMPI.hpp  --namespace=c10d --classname=WorkMPI > process-group-mpi.yaml

