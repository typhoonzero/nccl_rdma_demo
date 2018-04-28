CC=g++
INCLUDE_DIRS=-I/usr/local/cuda/include
LIBRARY_DIRS=-L/usr/local/cuda/lib64
CFLAGS=-std=c++11 $(INCLUDE_DIRS) $(LIBRARY_DIRS)
LDFLAGS=-lnccl -ldl -lcudart



all: demo

clean:
        rm -rf demo ./third_party/hiredis

hiredis:
        pushd ./third_party && git clone https://github.com/redis/hiredis.git && pushd hiredis && make && popd && popd

demo: nccl_rdma_demo.cc hiredis
        $(CC) $(CFLAGS) -o demo nccl_rdma_demo.cc ./third_party/hiredis/libhiredis.a $(LDFLAGS)