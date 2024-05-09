.. automodule:: dacapo

.. contents::
  :depth: 1
  :local:

Docker Configuration for JupyterHub-Dacapo
=========================================

This document provides instructions on how to build and run the Docker image for the JupyterHub-Dacapo project.

Requirements
------------
Before you begin, ensure you have Docker installed on your system. You can download it from `Docker's official website <https://docker.com>`_.

Building the Docker Image
-------------------------
To build the Docker image, navigate to the directory containing your Dockerfile and run the following command:

.. code-block:: bash

    docker build -t jupyterhub-dacapo .

This command builds a Docker image with the tag `jupyterhub-dacapo` using the Dockerfile in the current directory.

Running the Docker Container
----------------------------
Once the image is built, you can run a container from the image with the following command:

.. code-block:: bash

    docker run -p 8000:8000 jupyterhub-dacapo

This command starts a container based on the `jupyterhub-dacapo` image. It maps port 8000 of the container to port 8000 on the host, allowing you to access JupyterHub by navigating to `http://localhost:8000` in your web browser.

Stopping the Docker Container
-----------------------------
To stop the running container, you can use the Docker CLI to stop the container:

.. code-block:: bash

    docker stop [CONTAINER_ID]

Replace `[CONTAINER_ID]` with the actual ID of your running container. You can find the container ID by listing all running containers with `docker ps`.

Further Configuration
---------------------
For additional configuration options, such as setting environment variables or configuring volumes, refer to the Docker documentation or the specific documentation for the JupyterHub or Dacapo configurations.
