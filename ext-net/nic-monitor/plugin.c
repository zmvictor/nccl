/*************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

#define __hidden __attribute__ ((visibility("hidden")))
#define NCCL_PLUGIN_MAX_RECVS 1

// Maximum number of NICs we'll track
#define MAX_IB_DEVS 16

#define NIC_STATE_UNKNOWN 0
#define NIC_STATE_HEALTHY 1
#define NIC_STATE_FLAPPING 2
#define NIC_STATE_DOWN 3

typedef struct {
  struct ibv_context* context;
  char devName[64];
  uint8_t portNum;
  int state;
  int gidIndex;
  union ibv_gid gid;
  pthread_t asyncThread;
  int asyncThreadRunning;
} nicHealth_t;

int max_requests = NCCL_NET_MAX_REQUESTS;
static ncclDebugLogger_t logFunc = NULL;

static nicHealth_t nicState[MAX_IB_DEVS];
static int numDevices = 0;
static pthread_mutex_t nccl_ib_lock = PTHREAD_MUTEX_INITIALIZER;

#define WARN(...) if (logFunc) logFunc(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) if (logFunc) logFunc(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

static void* nicMonitorThread(void* args) {
  nicHealth_t* nic = (nicHealth_t*)args;
  INFO(NCCL_NET, "Starting monitor thread for device %s, port %d", nic->devName, nic->portNum);
  
  while (nic->asyncThreadRunning) {
    struct ibv_async_event event;
    if (ibv_get_async_event(nic->context, &event) == 0) {
      switch (event.event_type) {
        case IBV_EVENT_PORT_ACTIVE:
          INFO(NCCL_NET, "NIC %s port %d is now ACTIVE", nic->devName, nic->portNum);
          pthread_mutex_lock(&nccl_ib_lock);
          nic->state = NIC_STATE_HEALTHY;
          pthread_mutex_unlock(&nccl_ib_lock);
          
          if (ibv_query_gid(nic->context, nic->portNum, nic->gidIndex, &nic->gid) == 0) {
            INFO(NCCL_NET, "Updated GID for %s port %d at index %d", nic->devName, nic->portNum, nic->gidIndex);
          }
          break;
          
        case IBV_EVENT_PORT_ERR:
          WARN("NIC %s port %d is DOWN", nic->devName, nic->portNum);
          pthread_mutex_lock(&nccl_ib_lock);
          nic->state = NIC_STATE_DOWN;
          pthread_mutex_unlock(&nccl_ib_lock);
          break;
          
        case IBV_EVENT_GID_CHANGE:
          INFO(NCCL_NET, "GID changed for %s port %d", nic->devName, nic->portNum);
          if (ibv_query_gid(nic->context, nic->portNum, nic->gidIndex, &nic->gid) == 0) {
            INFO(NCCL_NET, "Updated GID after change for %s port %d at index %d", nic->devName, nic->portNum, nic->gidIndex);
          }
          break;
          
        default:
          INFO(NCCL_NET, "Received event %d for %s port %d", event.event_type, nic->devName, nic->portNum);
          break;
      }
      
      ibv_ack_async_event(&event);
    } else {
      usleep(10000); // 10ms
    }
  }
  
  INFO(NCCL_NET, "Exiting monitor thread for device %s, port %d", nic->devName, nic->portNum);
  return NULL;
}

__hidden ncclResult_t pluginInit(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  logFunc = logFunction;
  
  memset(nicState, 0, sizeof(nicState));
  
  struct ibv_device** devices;
  int ndev;
  
  devices = ibv_get_device_list(&ndev);
  if (!devices) {
    WARN("No IB devices found");
    return ncclSystemError;
  }
  
  for (int d=0; d<ndev && d<MAX_IB_DEVS; d++) {
    struct ibv_context* context = ibv_open_device(devices[d]);
    if (!context) {
      WARN("Failed to open device %s", ibv_get_device_name(devices[d]));
      continue;
    }
    
    struct ibv_device_attr devAttr;
    if (ibv_query_device(context, &devAttr) != 0) {
      WARN("Failed to query device %s", ibv_get_device_name(devices[d]));
      ibv_close_device(context);
      continue;
    }
    
    for (int p=1; p <= devAttr.phys_port_cnt; p++) {
      struct ibv_port_attr portAttr;
      if (ibv_query_port(context, p, &portAttr) != 0) {
        WARN("Failed to query port %d on device %s", p, ibv_get_device_name(devices[d]));
        continue;
      }
      
      if (portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) {
        continue;
      }
      
      nicState[numDevices].context = context;
      nicState[numDevices].portNum = p;
      strncpy(nicState[numDevices].devName, ibv_get_device_name(devices[d]), 63);
      nicState[numDevices].devName[63] = '\0';
      
      nicState[numDevices].gidIndex = 0;
      for (int g=0; g < portAttr.gid_tbl_len; g++) {
        if (ibv_query_gid(context, p, g, &nicState[numDevices].gid) == 0) {
          if (nicState[numDevices].gid.global.interface_id != 0 || 
              nicState[numDevices].gid.global.subnet_prefix != 0) {
            nicState[numDevices].gidIndex = g;
            break;
          }
        }
      }
      
      if (portAttr.state == IBV_PORT_ACTIVE) {
        nicState[numDevices].state = NIC_STATE_HEALTHY;
      } else {
        nicState[numDevices].state = NIC_STATE_DOWN;
      }
      
      nicState[numDevices].asyncThreadRunning = 1;
      if (pthread_create(&nicState[numDevices].asyncThread, NULL, 
                        nicMonitorThread, &nicState[numDevices]) != 0) {
        WARN("Failed to create async thread for %s port %d", 
              nicState[numDevices].devName, nicState[numDevices].portNum);
        nicState[numDevices].asyncThreadRunning = 0;
      }
      
      numDevices++;
    }
  }
  
  ibv_free_device_list(devices);
  
  INFO(NCCL_NET, "Initialized %d RoCE devices for monitoring", numDevices);
  return ncclSuccess;
}
__hidden ncclResult_t pluginDevices(int* ndev) {
  *ndev = numDevices;
  return ncclSuccess;
}
__hidden ncclResult_t pluginPciPath(int dev, char** path) { return ncclInternalError; }
__hidden ncclResult_t pluginPtrSupport(int dev, int* supportedTypes) { return ncclInternalError; }
static int getHealthyNicIndex(int dev) {
  pthread_mutex_lock(&nccl_ib_lock);
  
  if (dev < numDevices && nicState[dev].state == NIC_STATE_HEALTHY) {
    pthread_mutex_unlock(&nccl_ib_lock);
    return dev;
  }
  
  for (int i = 0; i < numDevices; i++) {
    if (nicState[i].state == NIC_STATE_HEALTHY) {
      pthread_mutex_unlock(&nccl_ib_lock);
      return i;
    }
  }
  
  pthread_mutex_unlock(&nccl_ib_lock);
  return -1; // No healthy NICs found
}

__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_t* props) {
  int healthyNic = getHealthyNicIndex(dev);
  if (healthyNic < 0) {
    WARN("No healthy NICs available");
    return ncclSystemError;
  }
  
  props->name = nicState[healthyNic].devName;
  props->pciPath = NULL;
  props->guid = healthyNic; // Use the index as the GUID for simplicity
  props->ptrSupport = NCCL_PTR_HOST;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = 100000; // 100Gbps - should be detected from port attributes in a real implementation
  props->port = nicState[healthyNic].portNum;
  props->latency = 0;
  props->maxComms = 1024*1024;
  props->maxRecvs = NCCL_PLUGIN_MAX_RECVS;
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = healthyNic;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  
  return ncclSuccess;
}

typedef struct {
  int dev;                       // Original device index
  int currentNic;                // Current healthy NIC index being used
  struct rdma_cm_id* cmId;       // RDMA connection manager ID
  struct ibv_qp* qp;             // Queue pair for this connection
  struct ibv_cq* cq;             // Completion queue
  int connected;                 // Connection status
  void* userData;                // User data for this connection
} pluginConnection_t;

typedef struct {
  int dev;                       // Device index
  struct rdma_event_channel* channel; // Event channel
  struct rdma_cm_id* cmId;       // Connection manager ID
  struct sockaddr_in addr;       // Local address
} pluginListenComm_t;

__hidden ncclResult_t pluginListen(int dev, void* opaqueHandle, void** listenComm) {
  int healthyNic = getHealthyNicIndex(dev);
  if (healthyNic < 0) {
    WARN("No healthy NICs available for listen");
    return ncclSystemError;
  }
  
  pluginListenComm_t* comm = (pluginListenComm_t*)calloc(1, sizeof(pluginListenComm_t));
  if (!comm) {
    WARN("Failed to allocate listen comm");
    return ncclSystemError;
  }
  
  comm->dev = healthyNic;
  
  comm->channel = rdma_create_event_channel();
  if (!comm->channel) {
    WARN("Failed to create RDMA event channel");
    free(comm);
    return ncclSystemError;
  }
  
  if (rdma_create_id(comm->channel, &comm->cmId, NULL, RDMA_PS_TCP) != 0) {
    WARN("Failed to create RDMA CM ID");
    rdma_destroy_event_channel(comm->channel);
    free(comm);
    return ncclSystemError;
  }
  
  if (rdma_bind_addr(comm->cmId, (struct sockaddr*)&comm->addr) != 0) {
    WARN("Failed to bind to local device");
    rdma_destroy_id(comm->cmId);
    rdma_destroy_event_channel(comm->channel);
    free(comm);
    return ncclSystemError;
  }
  
  if (rdma_listen(comm->cmId, 128) != 0) {
    WARN("Failed to start listening");
    rdma_destroy_id(comm->cmId);
    rdma_destroy_event_channel(comm->channel);
    free(comm);
    return ncclSystemError;
  }
  
  struct sockaddr* addr = rdma_get_local_addr(comm->cmId);
  memcpy(opaqueHandle, addr, sizeof(struct sockaddr_in));
  
  *listenComm = comm;
  return ncclSuccess;
}

__hidden ncclResult_t pluginConnect(int dev, ncclNetCommConfig_t* config, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  int healthyNic = getHealthyNicIndex(dev);
  if (healthyNic < 0) {
    WARN("No healthy NICs available for connect");
    return ncclSystemError;
  }
  
  pluginConnection_t* conn = (pluginConnection_t*)calloc(1, sizeof(pluginConnection_t));
  if (!conn) {
    WARN("Failed to allocate connection");
    return ncclSystemError;
  }
  
  conn->dev = dev;
  conn->currentNic = healthyNic;
  conn->connected = 0;
  
  struct rdma_event_channel* channel = rdma_create_event_channel();
  if (!channel) {
    WARN("Failed to create RDMA event channel");
    free(conn);
    return ncclSystemError;
  }
  
  if (rdma_create_id(channel, &conn->cmId, conn, RDMA_PS_TCP) != 0) {
    WARN("Failed to create RDMA CM ID");
    rdma_destroy_event_channel(channel);
    free(conn);
    return ncclSystemError;
  }
  
  if (rdma_bind_addr(conn->cmId, NULL) != 0) {
    WARN("Failed to bind to local device");
    rdma_destroy_id(conn->cmId);
    rdma_destroy_event_channel(channel);
    free(conn);
    return ncclSystemError;
  }
  
  struct sockaddr* remoteAddr = (struct sockaddr*)opaqueHandle;
  
  if (rdma_resolve_addr(conn->cmId, NULL, remoteAddr, 2000) != 0) {
    WARN("Failed to resolve remote address");
    rdma_destroy_id(conn->cmId);
    rdma_destroy_event_channel(channel);
    free(conn);
    return ncclSystemError;
  }
  
  struct rdma_cm_event* event;
  if (rdma_get_cm_event(channel, &event) == 0) {
    if (event->event == RDMA_CM_EVENT_ADDR_RESOLVED) {
      rdma_ack_cm_event(event);
      
      if (rdma_resolve_route(conn->cmId, 2000) != 0) {
        WARN("Failed to resolve route");
        rdma_destroy_id(conn->cmId);
        rdma_destroy_event_channel(channel);
        free(conn);
        return ncclSystemError;
      }
      
      if (rdma_get_cm_event(channel, &event) == 0) {
        if (event->event == RDMA_CM_EVENT_ROUTE_RESOLVED) {
          rdma_ack_cm_event(event);
          
          struct rdma_conn_param conn_param = {};
          if (rdma_connect(conn->cmId, &conn_param) != 0) {
            WARN("Connection in progress...");
            *sendComm = NULL;
            return ncclSuccess;
          }
        } else {
          rdma_ack_cm_event(event);
          WARN("Unexpected event during route resolution: %d", event->event);
          *sendComm = NULL;
          return ncclSuccess;
        }
      } else {
        WARN("Failed to get route resolution event");
        *sendComm = NULL;
        return ncclSuccess;
      }
    } else {
      rdma_ack_cm_event(event);
      WARN("Unexpected event during address resolution: %d", event->event);
      *sendComm = NULL;
      return ncclSuccess;
    }
  } else {
    WARN("Failed to get address resolution event");
    *sendComm = NULL;
    return ncclSuccess;
  }
  
  if (rdma_get_cm_event(channel, &event) == 0) {
    if (event->event == RDMA_CM_EVENT_ESTABLISHED) {
      rdma_ack_cm_event(event);
      conn->connected = 1;
      *sendComm = conn;
      
      if (sendDevComm) *sendDevComm = NULL; // We don't support device offload
      
      return ncclSuccess;
    }
    rdma_ack_cm_event(event);
  }
  
  *sendComm = NULL;
  return ncclSuccess;
}

__hidden ncclResult_t pluginAccept(void* listenCommOpaque, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
  pluginListenComm_t* listenComm = (pluginListenComm_t*)listenCommOpaque;
  
  struct rdma_cm_event* event;
  if (rdma_get_cm_event(listenComm->channel, &event) != 0) {
    *recvComm = NULL;
    return ncclSuccess;
  }
  
  if (event->event != RDMA_CM_EVENT_CONNECT_REQUEST) {
    rdma_ack_cm_event(event);
    *recvComm = NULL;
    return ncclSuccess;
  }
  
  pluginConnection_t* conn = (pluginConnection_t*)calloc(1, sizeof(pluginConnection_t));
  if (!conn) {
    WARN("Failed to allocate connection");
    rdma_ack_cm_event(event);
    return ncclSystemError;
  }
  
  conn->cmId = event->id;
  conn->dev = listenComm->dev;
  conn->currentNic = listenComm->dev; // Start with the same NIC
  conn->connected = 1;
  
  rdma_accept(conn->cmId, NULL);
  rdma_ack_cm_event(event);
  
  *recvComm = conn;
  if (recvDevComm) *recvDevComm = NULL; // We don't support device offload
  
  return ncclSuccess;
}
typedef struct {
  void* addr;
  size_t size;
  int type;
} pluginMr_t;

__hidden ncclResult_t pluginRegMr(void* commOpaque, void* data, size_t size, int type, void** mhandle) {
  // Validate connection
  pluginConnection_t* comm = (pluginConnection_t*)commOpaque;
  if (!comm) {
    WARN("Invalid connection for regMr");
    return ncclInternalError;
  }
  
  pluginMr_t* mr = (pluginMr_t*)calloc(1, sizeof(pluginMr_t));
  if (!mr) {
    WARN("Failed to allocate memory registration");
    return ncclSystemError;
  }
  
  mr->addr = data;
  mr->size = size;
  mr->type = type;
  
  *mhandle = mr;
  return ncclSuccess;
}

__hidden ncclResult_t pluginRegMrDmaBuf(void* commOpaque, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  return ncclInternalError;
}

__hidden ncclResult_t pluginDeregMr(void* commOpaque, void* mhandle) {
  pluginMr_t* mr = (pluginMr_t*)mhandle;
  free(mr);
  return ncclSuccess;
}
typedef struct {
  void* data;
  size_t size;
  int tag;
  void* mhandle;
  int done;
  pluginConnection_t* comm;
} pluginRequest_t;

__hidden ncclResult_t pluginIsend(void* sendCommOpaque, void* data, size_t size, int tag, void* mhandle, void* phandle, void** requestOpaque) {
  pluginConnection_t* comm = (pluginConnection_t*)sendCommOpaque;
  
  pthread_mutex_lock(&nccl_ib_lock);
  if (nicState[comm->currentNic].state != NIC_STATE_HEALTHY) {
    int newNic = -1;
    for (int i = 0; i < numDevices; i++) {
      if (nicState[i].state == NIC_STATE_HEALTHY) {
        newNic = i;
        break;
      }
    }
    
    if (newNic >= 0) {
      INFO(NCCL_NET, "Switching send from NIC %d to NIC %d", comm->currentNic, newNic);
      comm->currentNic = newNic;
    } else {
      WARN("No healthy NICs available for send");
      pthread_mutex_unlock(&nccl_ib_lock);
      return ncclSystemError;
    }
  }
  pthread_mutex_unlock(&nccl_ib_lock);
  
  pluginRequest_t* req = (pluginRequest_t*)calloc(1, sizeof(pluginRequest_t));
  if (!req) {
    WARN("Failed to allocate request");
    return ncclSystemError;
  }
  
  req->data = data;
  req->size = size;
  req->tag = tag;
  req->mhandle = mhandle;
  req->done = 0;
  req->comm = comm;
  
  req->done = 1;
  
  *requestOpaque = req;
  return ncclSuccess;
}

__hidden ncclResult_t pluginIrecv(void* recvCommOpaque, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** requestOpaque) {
  if (n != 1) {
    WARN("This plugin doesn't support multiple receives");
    return ncclInternalError;
  }
  
  pluginConnection_t* comm = (pluginConnection_t*)recvCommOpaque;
  
  pthread_mutex_lock(&nccl_ib_lock);
  if (nicState[comm->currentNic].state != NIC_STATE_HEALTHY) {
    int newNic = -1;
    for (int i = 0; i < numDevices; i++) {
      if (nicState[i].state == NIC_STATE_HEALTHY) {
        newNic = i;
        break;
      }
    }
    
    if (newNic >= 0) {
      INFO(NCCL_NET, "Switching recv from NIC %d to NIC %d", comm->currentNic, newNic);
      comm->currentNic = newNic;
    } else {
      WARN("No healthy NICs available for recv");
      pthread_mutex_unlock(&nccl_ib_lock);
      return ncclSystemError;
    }
  }
  pthread_mutex_unlock(&nccl_ib_lock);
  
  pluginRequest_t* req = (pluginRequest_t*)calloc(1, sizeof(pluginRequest_t));
  if (!req) {
    WARN("Failed to allocate request");
    return ncclSystemError;
  }
  
  req->data = data[0];
  req->size = sizes[0];
  req->tag = tags[0];
  req->mhandle = mhandles[0];
  req->done = 0;
  req->comm = comm;
  
  req->done = 1;
  
  *requestOpaque = req;
  return ncclSuccess;
}

__hidden ncclResult_t pluginIflush(void* recvCommOpaque, int n, void** data, int* sizes, void** mhandles, void** requestOpaque) {
  pluginRequest_t* req = (pluginRequest_t*)calloc(1, sizeof(pluginRequest_t));
  if (!req) {
    WARN("Failed to allocate request");
    return ncclSystemError;
  }
  
  req->done = 1;
  
  *requestOpaque = req;
  return ncclSuccess;
}

__hidden ncclResult_t pluginTest(void* requestOpaque, int* done, int* sizes) {
  pluginRequest_t* req = (pluginRequest_t*)requestOpaque;
  
  *done = req->done;
  
  if (*done && sizes) {
    sizes[0] = req->size;
  }
  
  return ncclSuccess;
}
__hidden ncclResult_t pluginCloseSend(void* sendCommOpaque) {
  pluginConnection_t* comm = (pluginConnection_t*)sendCommOpaque;
  
  if (comm) {
    if (comm->cmId) {
      rdma_disconnect(comm->cmId);
      rdma_destroy_id(comm->cmId);
    }
    
    free(comm);
  }
  
  return ncclSuccess;
}

__hidden ncclResult_t pluginCloseRecv(void* recvCommOpaque) {
  pluginConnection_t* comm = (pluginConnection_t*)recvCommOpaque;
  
  if (comm) {
    if (comm->cmId) {
      rdma_disconnect(comm->cmId);
      rdma_destroy_id(comm->cmId);
    }
    
    free(comm);
  }
  
  return ncclSuccess;
}

__hidden ncclResult_t pluginCloseListen(void* listenCommOpaque) {
  pluginListenComm_t* comm = (pluginListenComm_t*)listenCommOpaque;
  
  if (comm) {
    if (comm->cmId) {
      rdma_destroy_id(comm->cmId);
    }
    
    if (comm->channel) {
      rdma_destroy_event_channel(comm->channel);
    }
    
    free(comm);
  }
  
  return ncclSuccess;
}
__hidden ncclResult_t pluginIrecvConsumed(void* recvCommOpaque, int n, void* requestOpaque) {
  // We don't support device offload
  return ncclInternalError;
}

__hidden ncclResult_t pluginGetDeviceMr(void* commOpaque, void* mhandle, void** dptr_mhandle) {
  // We don't support device offload
  return ncclInternalError;
}

__hidden ncclResult_t pluginMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  // We don't support virtual devices
  return ncclInternalError;
}

#define PLUGIN_NAME "NIC-Monitor"

__attribute__((destructor)) static void pluginCleanup() {
  for (int i = 0; i < numDevices; i++) {
    if (nicState[i].asyncThreadRunning) {
      nicState[i].asyncThreadRunning = 0;
      pthread_join(nicState[i].asyncThread, NULL);
    }
    
    if (nicState[i].context) {
      ibv_close_device(nicState[i].context);
      nicState[i].context = NULL;
    }
  }
  
  numDevices = 0;
}

const ncclNet_v10_t ncclNetPlugin_v10 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties,
  .listen = pluginListen,
  .connect = pluginConnect,
  .accept = pluginAccept,
  .regMr = pluginRegMr,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend,
  .irecv = pluginIrecv,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
  .getDeviceMr = pluginGetDeviceMr,
  .irecvConsumed = pluginIrecvConsumed,
  .makeVDevice   = pluginMakeVDevice,
};

__hidden ncclResult_t pluginInit_v9(ncclDebugLogger_t logFunction) {
  return pluginInit(logFunction, NULL);
}

__hidden ncclResult_t pluginGetProperties_v9(int dev, ncclNetProperties_v9_t* props) {
  return pluginGetProperties(dev, (ncclNetProperties_t*)props);
}

__hidden ncclResult_t pluginConnect_v9(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm){
  return pluginConnect(dev, NULL, handle, sendComm, sendDevComm);
}

__hidden ncclResult_t pluginIsend_v9(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
  return pluginIsend(sendComm, data, size, tag, mhandle, NULL, request);
}

__hidden ncclResult_t pluginIrecv_v9(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
  return pluginIrecv(recvComm, n, data, sizes, tags, mhandles, NULL, request);
}

__hidden ncclResult_t pluginMakeVDevice_v9(int* d, ncclNetVDeviceProps_v9_t* props) { return ncclInternalError; }

const ncclNet_v9_t ncclNetPlugin_v9 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v9,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v9,
  .listen = pluginListen,
  .connect = pluginConnect_v9,
  .accept = pluginAccept,
  .regMr = pluginRegMr,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v9,
  .irecv = pluginIrecv_v9,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
  .getDeviceMr = pluginGetDeviceMr,
  .irecvConsumed = pluginIrecvConsumed,
  .makeVDevice   = pluginMakeVDevice_v9,
};

__hidden ncclResult_t pluginGetProperties_v8(int dev, ncclNetProperties_v8_t* props_v8) {
  ncclNetProperties_t props;
  ncclResult_t ret = pluginGetProperties(dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v8->name = props.name;
  props_v8->pciPath = props.pciPath;
  props_v8->guid = props.guid;
  props_v8->ptrSupport = props.ptrSupport;
  props_v8->regIsGlobal = props.regIsGlobal;
  props_v8->speed = props.speed;
  props_v8->latency = props.latency;
  props_v8->port = props.port;
  props_v8->maxComms = props.maxComms;
  props_v8->maxRecvs = props.maxRecvs;
  props_v8->netDeviceType = props.netDeviceType;
  props_v8->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

__hidden ncclResult_t pluginIsend_v8(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  return pluginIsend(sendComm, data, (int)size, tag, mhandle, NULL, request);
}

__hidden ncclResult_t pluginIrecv_v8(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  size_t sizesOut[NCCL_PLUGIN_MAX_RECVS];
  for (int i=0; i<n; i++) sizesOut[i] = sizes[i];
  return pluginIrecv(recvComm, 1, data, sizesOut, tags, mhandles, NULL, request);
}

const ncclNet_v8_t ncclNetPlugin_v8 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v9,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v8,
  .listen = pluginListen,
  .connect = pluginConnect_v9,
  .accept = pluginAccept,
  .regMr = pluginRegMr,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v8,
  .irecv = pluginIrecv_v8,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
  .getDeviceMr = pluginGetDeviceMr,
  .irecvConsumed = pluginIrecvConsumed,
 };

__hidden ncclResult_t pluginGetProperties_v7(int dev, ncclNetProperties_v7_t* props_v7) {
  ncclNetProperties_t props;
  ncclResult_t ret = pluginGetProperties(dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v7->name = props.name;
  props_v7->pciPath = props.pciPath;
  props_v7->guid = props.guid;
  props_v7->ptrSupport = props.ptrSupport;
  props_v7->speed = props.speed;
  props_v7->latency = props.latency;
  props_v7->port = props.port;
  props_v7->maxComms = props.maxComms;
  props_v7->maxRecvs = props.maxRecvs;
  props_v7->netDeviceType = props.netDeviceType;
  props_v7->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

__hidden ncclResult_t pluginRegMr_v7(void* collComm, void* data, int size, int type, void** mhandle) {
  return pluginRegMr(collComm, data, size, type, mhandle);
}

const ncclNet_v7_t ncclNetPlugin_v7 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v9,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v7,
  .listen = pluginListen,
  .connect = pluginConnect_v9,
  .accept = pluginAccept,
  .regMr = pluginRegMr_v7,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v8,
  .irecv = pluginIrecv_v8,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
  .getDeviceMr = pluginGetDeviceMr,
  .irecvConsumed = pluginIrecvConsumed,
};

__hidden ncclResult_t pluginGetProperties_v6(int dev, ncclNetProperties_v6_t* props_v6) {
  ncclNetProperties_t props;
  ncclResult_t ret = pluginGetProperties(dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v6->name = props.name;
  props_v6->pciPath = props.pciPath;
  props_v6->guid = props.guid;
  props_v6->ptrSupport = props.ptrSupport;
  props_v6->speed = props.speed;
  props_v6->latency = props.latency;
  props_v6->port = props.port;
  props_v6->maxComms = props.maxComms;
  props_v6->maxRecvs = props.maxRecvs;
  return ncclSuccess;
}

__hidden ncclResult_t pluginConnect_v6(int dev, void* handle, void** sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginAccept_v6(void* listenComm, void** recvComm) { return ncclInternalError; }

const ncclNet_v6_t ncclNetPlugin_v6 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v9,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v6,
  .listen = pluginListen,
  .connect = pluginConnect_v6,
  .accept = pluginAccept_v6,
  .regMr = pluginRegMr_v7,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v8,
  .irecv = pluginIrecv_v8,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen
};

/* v5 Compat */
const ncclNet_v5_t ncclNetPlugin_v5 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v9,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v6,
  .listen = pluginListen,
  .connect = pluginConnect_v6,
  .accept = pluginAccept_v6,
  .regMr = pluginRegMr_v7,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v8,
  .irecv = pluginIrecv_v8,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
};

/* v4 Compat */
static ncclResult_t pluginGetProperties_v4(int dev, ncclNetProperties_v4_t* props_v4) {
  ncclNetProperties_t props;
  ncclResult_t ret = pluginGetProperties(dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v4->name = props.name;
  props_v4->pciPath = props.pciPath;
  props_v4->guid = props.guid;
  props_v4->ptrSupport = props.ptrSupport;
  props_v4->speed = props.speed;
  props_v4->port = props.port;
  props_v4->maxComms = props.maxComms;
  return ncclSuccess;
}
static ncclResult_t pluginIsend_v4(void *sendComm, void* data, int size, void *mhandle, void** request) {
  return pluginIsend_v8(sendComm, data, size, 0, mhandle, request);
}
static ncclResult_t pluginIrecv_v4(void* recvComm, void* data, int size, void* mhandle, void** request) {
  int tag = 0;
  return pluginIrecv_v8(recvComm, 1, &data, &size, &tag, &mhandle, request);
}
static ncclResult_t pluginIflush_v4(void* recvComm, void* data, int size, void* mhandle, void** request) {
  return pluginIflush(recvComm, 1, &data, &size, &mhandle, request);
}
static ncclResult_t pluginConnect_v4(int dev, void* handle, void** sendComm) {
  ncclResult_t ret;
  do {
    ncclNetDeviceHandle_v7_t* handle = NULL;
    ret = pluginConnect(dev, NULL, handle, sendComm, &handle);
  } while (ret == ncclSuccess && *sendComm == NULL);
  return ret;
}
static ncclResult_t pluginAccept_v4(void* listenComm, void** recvComm) {
  ncclResult_t ret;
  do {
    ncclNetDeviceHandle_v7_t* handle = NULL;
    ret = pluginAccept(listenComm, recvComm, &handle);
  } while (ret == ncclSuccess && *recvComm == NULL);
  return ret;
}
const ncclNet_v4_t ncclNetPlugin_v4 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v9,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v4,
  .listen = pluginListen,
  .connect = pluginConnect_v4,
  .accept = pluginAccept_v4,
  .regMr = pluginRegMr_v7,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v4,
  .irecv = pluginIrecv_v4,
  .iflush = pluginIflush_v4,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
};

/* v3 Compat */
static ncclResult_t pluginFlush(void* recvComm, void* data, int size, void* mhandle) {
  void* req;
  ncclResult_t ret = pluginIflush_v4(recvComm, data, size, mhandle, &req);
  int done = 0;
  while (ret == ncclSuccess && done == 0) {
    ret = pluginTest(req, &done, NULL);
  }
  return ret;
}
static ncclResult_t pluginInit_v3(ncclDebugLogger_t logFunction) {
  max_requests = NCCL_NET_MAX_REQUESTS_V3;
  return pluginInit(logFunction, NULL);
}
#include <string.h>
static ncclResult_t pluginListen_v3(int dev, void* handle, void** listenComm) {
  char pluginHandle[NCCL_NET_HANDLE_MAXSIZE];
  ncclResult_t ret = pluginListen(dev, &pluginHandle, listenComm);
  memcpy(handle, &pluginHandle, NCCL_NET_HANDLE_MAXSIZE_V4);
  return ret;
}
static ncclResult_t pluginConnect_v3(int dev, void* handle, void** sendComm) {
  char pluginHandle[NCCL_NET_HANDLE_MAXSIZE];
  memcpy(&pluginHandle, handle, NCCL_NET_HANDLE_MAXSIZE_V4);
  return pluginConnect_v4(dev, &pluginHandle, sendComm);
}
const ncclNet_v3_t ncclNetPlugin_v3 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v3,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v4,
  .listen = pluginListen_v3,
  .connect = pluginConnect_v3,
  .accept = pluginAccept_v4,
  .regMr = pluginRegMr_v7,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v4,
  .irecv = pluginIrecv_v4,
  .flush = pluginFlush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
};

/* v2 Compat */
const ncclNet_v2_t ncclNetPlugin_v2 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v3,
  .devices = pluginDevices,
  .pciPath = pluginPciPath,
  .ptrSupport = pluginPtrSupport,
  .listen = pluginListen,
  .connect = pluginConnect_v4,
  .accept = pluginAccept_v4,
  .regMr = pluginRegMr_v7,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v4,
  .irecv = pluginIrecv_v4,
  .flush = pluginFlush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
};
