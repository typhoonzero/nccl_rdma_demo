#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <string.h>

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

  bool SetId(const ncclUniqueId& uniq_id) {
    std::vector<const char*> command;
    std::vector<size_t> lens;
    bool ret = false;

    auto cmd1 = std::string("SET");
    command.push_back(cmd1.c_str());
    lens.push_back(cmd1.size());

    command.push_back(id_key_.c_str());
    lens.push_back(id_key_.size());

    auto cmd3 = std::string(uniq_id.internal, NCCL_UNIQUE_ID_BYTES);
    command.push_back(cmd3.c_str());
    lens.push_back(NCCL_UNIQUE_ID_BYTES);

    redisReply *reply = static_cast<redisReply*>(
      redisCommandArgv(ctx_, command.size(), command.data(), lens.data()));
    if (reply->type == REDIS_REPLY_STATUS) {
      auto ret_status = std::string(reply->str, reply->len);
      if (ret_status == "OK") {
        ret = true;
      }
    } else if (reply->type == REDIS_REPLY_ERROR) {
      std::cerr << "error: " << reply->str << std::endl;
    } else {
      std::cerr << "error response type: " << reply->type << std::endl;
    }
    freeReplyObject(reply);

    return ret;
  }

ncclUniqueId GetClusterId(int node_count) {
    ncclUniqueId id;

    while(!Exists()) {
      sleep(1);
    }

    redisReply *reply = static_cast<redisReply*>(
      redisCommand(ctx_, "GET %s", id_key_.c_str()));

    if (reply->type == REDIS_REPLY_STRING) {
      ncclUniqueId id;
      if (reply->len != NCCL_UNIQUE_ID_BYTES) {
        std::cerr << "redis reply uniq id data error" << std::endl;
        exit(1);
      }
      memcpy(id.internal, reply->str, reply->len);
    }
    freeReplyObject(reply);
    return id;
  }

 private:
  bool Exists() {
    bool ret = false;
    redisReply *reply = static_cast<redisReply*>(
        redisCommand(ctx_, "EXISTS %s", id_key_.c_str()));
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
  const std::string id_key_ = "nccl_id_key";
};
