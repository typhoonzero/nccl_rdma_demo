demo: nccl_rdma_demo.cc
	g++ -std=c++11 -o demo nccl_rdma_demo.cc ./third_party/hiredis/libhiredis.a
