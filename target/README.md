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
- 
**Troubleshooting**
It might happen that the latest docker image ``mcr.microsoft.
com/azure-functions/python:4-python3.x`` is newer than the images that are 
running in the productive environment in Azure. Thus a library compiled 
against the younger version may not be executable when deployed to Azure:

```
Result: Failure Exception: 
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found 
(required by /home/site/wwwroot/.python_packages/lib/site-packages/jpt/distributions/quantile/cdfreg.cpython-38-x86_64-linux-gnu.so). 
Please check the requirements.txt file for the missing module. 
For more info, please refer the troubleshooting guide: https://aka.ms/functions-modulenotfound 
Stack: File "/azure-functions-host/workers/python/3.8/LINUX/X64/azure_funct...
```
This has indeed unrelated to the ``requirements.txt`` but with the system 
libraries in the host system. In such a case, you can try using an older version
of the docker image found in [this list](https://mcr.microsoft.com/v2/azure-functions/python/tags/list).
