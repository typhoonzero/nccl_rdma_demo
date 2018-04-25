#include "nccl.h"
#include "redis_client.h"

void ncclCommInitAllNodes(ncclComm_t* comm,
                                  int ndev,
                                  const int* devlist,
                                  const ncclUniqueId& id) {
  ncclGroupStart();
  for (int i=0; i<ndev; i++) {
    cudaSetDevice(devlist[i]);
    ncclCommInitRank(comm+i, ndev, id, i);
  }
  ncclGroupEnd();
}

int main(int argc, char** argv) {
  if (argc < 1) {
    std::cerr << "must provide cluster total trainer count" << std::endl;
  }
  int node_count = atoi([argv[0]]);
  NcclIdServer id_server("127.0.0.1", 6379);

  ncclUniqueId id;
  ncclGetUniqueId(&id);

  id_server.SetIds(0, id);
  auto all_ids = id_server.GetClusterIds(node_count);
  ncclComm_t comm[node_count * 4];  // 4 GPU
  int devlist[4] = {0,1,2,3};
  for ()
    ncclCommInitAllNodes(comm, 4, devlist, );
  }
}
