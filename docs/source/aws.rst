.. automodule:: dacapo

.. contents::
  :depth: 1
  :local:

AWS EC2 Setup Guide
===================

This guide will help you to run your Docker image on an AWS EC2 instance and set up S3 access for storing data.

Running Docker Image on AWS EC2
-------------------------------

To run your Docker image on an AWS EC2 instance, follow these steps:

1. **Create Key Pair** (if you don't have one already):

.. code-block:: bash

   aws ec2 create-key-pair --key-name MyKeyPair --query 'KeyMaterial' --output text > MyKeyPair.pem
   chmod 400 MyKeyPair.pem

2. **Create a Security Group** (if you don't have one already):

.. code-block:: bash

   aws ec2 create-security-group --group-name my-security-group --description "My security group"


3. **Authorize Inbound Traffic for the Security Group**:

.. code-block:: bash
   aws ec2 authorize-security-group-ingress --group-name my-security-group --protocol tcp --port 22 --cidr 0.0.0.0/0
   aws ec2 authorize-security-group-ingress --group-name my-security-group --protocol tcp --port 80 --cidr 0.0.0.0/0


4. **Run EC2 Instance with Docker Image**:

   Use the following script to launch an EC2 instance, pull your Docker image from DockerHub, and run it with port forwarding:

.. code-block:: bash

   aws ec2 run-instances      --image-id ami-0b8956f13d7bdfe7b      --count 1      --instance-type p3.2xlarge      --key-name <YOUR_KEY_PAIR>      --security-groups <YOUR_SECURITY_GROUP>      --user-data '#!/bin/bash
     yum update -y
     amazon-linux-extras install docker -y
     service docker start
     docker pull mzouink/dacapo:0.3.0
     docker run -d -p 80:8000 mzouink/dacapo:0.3.0'


Replace `<YOUR_KEY_PAIR>` with the name of your key pair and `<YOUR_SECURITY_GROUP>` with the name of your security group.

S3 Access Configuration
-----------------------

You can work locally using S3 data by setting the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables. You can also set the `AWS_REGION` environment variable to specify the region to use. If you are using a profile, you can set the `AWS_PROFILE` environment variable to specify the profile to use.

To configure AWS CLI, use the following command:

.. code-block:: bash

    aws configure


Storing Checkpoints and Experiments Data in S3
----------------------------------------------

To store checkpoints and experiments data in S3, modify the `dacapo.yaml` file to include the following:

.. code-block:: yaml
    
    runs_base_dir: "s3://dacapotest"


This setup will help you run your Docker image on AWS EC2 and configure S3 access for storing checkpoints and experiment data.
