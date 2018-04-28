# NCCL multi node communication demo

Some demo code to show how to use NCCL2 to do AllReduce on multiple nodes.

# Build

To build, you **must** install CUDA and NCCL before start to build:

```bash
git clone https://github.com/typhoonzero/nccl_rdma_demo.git
cd nccl_rdma_demo
make
```

# Run

We use [Redis](https://redis.io) to broadcast NCCL unique id, so start a redis
instance before you start, if you have docker, just run:

```bash
docker run -d --name myredis -p 6379:6379 redis
```

Run `./demo` to get help message.

```
Usage: demo [redisip:port] [node count] [node id]
```

Assume you have 2 nodes, each node have 4 GPUs, then you can run the demo like below:

* On node 1: `./demo [redis ip]:6379 2 0`
* On node 2: `./demo [redis ip]:6379 2 1`
