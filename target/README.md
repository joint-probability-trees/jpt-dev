# Build Targets

## Azure Functions
**Requirements**
- docker
 
To be able to run a jpt wheel in an Azure Function, its dynamically linked library dependencies must be compatible with 
those of the targeted function worker.  

We have opted for the pragmatic solution of running the jpt compilation process inside an Azure Function worker process, 
modified for compilation.

Run all of the following  commands from the jpt root directory

To build the worker docker container, run

```shell
docker build -t azure-function-build-worker -f target/azure-function-builder.dockerfile .
```  

To perform a build, run
```shell
docker run --rm -v $(pwd):/app azure-function-build-worker
```

Please note that:
- The artifact will be compiled to the dist folder
- Since it is built by docker, it will belong the system's root user, so file ownership and permission changes might
be necessary. 
