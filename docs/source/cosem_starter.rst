
Fine-Tune Cosem Starter
============================

The CosemStarter in DaCapo allows you to load a pretrained COSEM model and fine-tune it for your experiments. This guide explains how to set up and use CosemStarter in DaCapo.

Prerequisites
-------------

Ensure that you have DaCapo installed and configured correctly.

Step 1: Import the CosemStartConfig
-----------------------------------

To get started, you need to import `CosemStartConfig` from `dacapo.experiments.starts`.

.. code-block:: python

    from dacapo.experiments.starts import CosemStartConfig

Step 2: Configure the Start Model
---------------------------------

The `CosemStartConfig` takes two parameters:

- **model_name**: The name of the model setup to load.
- **checkpoint**: The specific checkpoint ID to load the pretrained model from.

Example:

.. code-block:: python

    # We will now download a pretrained COSEM model and fine-tune from that model.
    # It will only download the model the first time it is used.

    start_config = CosemStartConfig("setup04", "1820500")

This configuration will download the COSEM model from setup `setup04` and load the checkpoint `1820500`. You only need to download the model once; subsequent runs will use the downloaded model.

Step 3: Create a Run with `start_config`
----------------------------------------

To start from the pretrained model, add `start_config` to your `RunConfig`. The `RunConfig` initializes the run and allows fine-tuning from the pretrained COSEM model.

Example:

.. code-block:: python

    from dacapo.experiments.runs import RunConfig

    run_config = RunConfig(
        # other parameters...
        start_config=start_config,
    )

Full Example
------------

Here's how the complete setup looks:

.. code-block:: python

    from dacapo.experiments.starts import CosemStartConfig
    from dacapo.experiments.runs import RunConfig

    # Define the start configuration to load the pretrained COSEM model
    start_config = CosemStartConfig("setup04", "1820500")

    # Define the run configuration with the start configuration
    run_config = RunConfig(
        # other configurations,
        start_config=start_config,
    )

    # Now you can run this configuration in your experiment to start from the COSEM pretrained model

This setup will initiate your DaCapo run from the pretrained COSEM model and allow you to fine-tune it as needed.

Available COSEM Pretrained Models
---------------------------------

Below is a table of the COSEM pretrained models available, along with their details:

.. list-table:: Available COSEM Pretrained Models
    :header-rows: 1

    * - Model
      - Checkpoints
      - Best Checkpoint
      - Classes
      - Input Res
      - Output Res
      - Model
    * - setup04
      - 975000, 625000, 1820500
      - 1820500
      - ecs, pm, mito, mito_mem, ves, ves_mem, endo, endo_mem, er, er_mem, eres, nuc, mt, mt_out
      - 8 nm
      - 4 nm
      - Upsample U-Net
    * - setup26.1
      - 650000, 2580000
      - 2580000
      - mito, mito_mem, mito_ribo
      - 8 nm
      - 4 nm
      - Upsample U-Net
    * - setup28
      - 775000
      - 775000
      - er, er_mem
      - 8 nm
      - 4 nm
      - Upsample U-Net
    * - setup36
      - 500000, 1100000
      - 1100000
      - nuc, nucleo
      - 8 nm
      - 4 nm
      - Upsample U-Net
    * - setup45
      - 625000, 1634500
      - 1634500
      - ecs, pm
      - 4 nm
      - 4 nm
      - U-Net


Notes
-----

- The model will download only the first time you use it. After that, it will reuse the downloaded version.
- Ensure that you have the necessary storage and access permissions configured for the COSEM model files.
