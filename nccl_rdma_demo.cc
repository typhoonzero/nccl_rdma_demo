#include <sstream>
#include <cassert>
#include <stdio.h>

#include "nccl.h"
#include "redis_client.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


/*
 * This is an example of using nccl to do multi node multi GPU communication
 * without using MPI, in additional to the original example at:
 * https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#multidevprothrd
*/

std::vector<std::string> split(const std::string &s, char delim) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> tokens;
  while (getline(ss, item, delim)) {
      tokens.push_back(item);
  }
  return tokens;
}

void InitMemory(int gpu_count, int node_id, int size,
                float** sendbuff, float** recvbuff,
                cudaStream_t* s) {
  for (int i = 0; i < gpu_count; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Usage: demo [redisip:port] [node count] [node id]" << std::endl;
    exit(1);
  }
  std::string endpoint(argv[1]);
  auto ep_parts = split(endpoint, ':');
  std::cout << "parts: " << ep_parts.size()
            << " endpoints: " << endpoint << std::endl;

  assert(ep_parts.size() == 2);

  int node_count = atoi(argv[2]);
  int node_id = atoi(argv[3]);

  NcclIdServer id_server(ep_parts[0], atoi(ep_parts[1].c_str()));

  ncclUniqueId id;
  if (node_id == 0) {
    ncclGetUniqueId(&id);
    id_server.SetId(id);
  }
  auto nccl_id = id_server.GetClusterId(node_count);

  int gpu_count = 4;
  ncclComm_t comms[gpu_count];  // 4 GPU
  int devlist[4] = {0,1,2,3};
  float** sendbuff = (float**)malloc(gpu_count * sizeof(float*));
  float** recvbuff = (float**)malloc(gpu_count * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*gpu_count);

  // alloc GPU memory and set sendbuff to value 1.
  InitMemory(gpu_count, node_id, 4096, sendbuff, recvbuff, s);

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < gpu_count; ++i) {
    int rank = node_id * gpu_count + i;
    int nranks = node_count * gpu_count;
    CUDACHECK(cudaSetDevice(i));
    std::cout << "set gpu " << i << " rank " << rank
              << " nranks " << nranks << std::endl;
    NCCLCHECK(ncclCommInitRank(comms+i, nranks, nccl_id, rank));
  }
  NCCLCHECK(ncclGroupEnd());

  // do allreduce
  std::cout << "start allreduce call..." << std::endl;  
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < gpu_count; i++)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i],
        4096, ncclFloat, ncclSum,
        comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());
  std::cout << "end allreduce call..." << std::endl;

  for (int i = 0; i < gpu_count; i++) {
    ncclCommDestroy(comms[i]);
  }
}
