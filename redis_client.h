#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>

#include "third_party/hiredis/hiredis.h"
#include "nccl.h"

class NcclIdServer {

 public:
  NcclIdServer(const std::string& ip, int port) :
    ip_(ip), port_(port) {
    ctx_ = redisConnect(ip_.c_str(), port_);
    if (ctx_ == nullptr || ctx_->err) {
      if (ctx_) {
        std::cerr << "Error: " << ctx_->errstr << std::endl;
      } else {
        std::cerr << "Can't allocate redis context" << std::endl;
      }
    }
  }

  ~NcclIdServer() { redisFree(ctx_); }

  bool SetIds(int trainer_id, const ncclUniqueId& uniq_id) {
    char commands[3][NCCL_UNIQUE_ID_BYTES];
    size_t lens[3];
    bool ret = false;
    for (auto &per_id : ids) {
      command[0] = "SET";
      lens[0] = strlen(command[0]);
      sprintf(command[1], "trainer:%03d", trainer_id);
      lens[1] = strlen(command[1]);
      memcpy(command[2], uniq_id.internal, NCCL_UNIQUE_ID_BYTES);
      lens[3] = NCCL_UNIQUE_ID_BYTES;
      redisReply *reply = static_cast<redisReply*>(
        redisCommandArgv(ctx_, 3, command, lens));
      if (reply->type == REDIS_REPLY_STATUS) {
        ret_status = std::string(reply->str, reply->len);
        if (ret_status == "OK") {
          ret = true;
        }
      } else {
        std::cerr << "error response type" << std::endl;
      }
      freeReplyObject(reply);
    }
    return ret;
  }

  std::vector<ncclUniqueId> GetClusterIds(int node_count) {
    std::vector<ncclUniqueId> ret;
    for (int i = 0; i < node_count; ++i) {
      while(!Exists(i)) {
        sleep(1);
      }
      redisReply *reply = static_cast<redisReply*>(
        redisCommand(ctx_, "GET trainer:%03d", i));
      if (reply->type == REDIS_REPLY_STRING) {
        for (int i = 0; i < reply->elements; ++i) {
          if (per_reply->type == REDIS_REPLY_STRING) {
            ncclUniqueId id;
            if (reply->len != NCCL_UNIQUE_ID_BYTES) {
              std::cerr << "redis reply uniq id data error" << std::endl;
            }
            memcpy(id.internal, reply->str, reply->len);
            ret.emplace_back(id);
          }
        }
      }
      freeReplyObject(reply);
    }
    return ret;
  }

 private:
  bool Exists(int trainer_id) {
    char command[512];
    bool ret = false;
    sprintf(command, "EXISTS trainer:%03d", trainer_id);
    redisReply *reply = static_cast<redisReply*>(
        redisCommand(ctx_, command));
    if (reply->type == REDIS_REPLY_INTEGER) {
      if (reply->integer == 1) {
        ret = true;
      }
    }
    freeReplyObject(reply);
    return ret;
  }

 private:
  std::string ip_;
  int port_;
  redisContext *ctx_ = nullptr;
};
