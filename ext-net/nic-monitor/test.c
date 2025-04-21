#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

int main(int argc, char** argv) {
  printf("Testing NIC monitoring plugin for NCCL\n");
  
  setenv("NCCL_DEBUG", "INFO", 1);
  
  void* handle = dlopen("./libnccl-net-nic-monitor.so", RTLD_NOW);
  if (!handle) {
    fprintf(stderr, "Error loading plugin: %s\n", dlerror());
    return 1;
  }
  
  printf("Successfully loaded NIC monitoring plugin\n");
  
  printf("Sleeping for 10 seconds to allow monitoring...\n");
  sleep(10);
  
  dlclose(handle);
  printf("Plugin test completed successfully\n");
  
  return 0;
}
