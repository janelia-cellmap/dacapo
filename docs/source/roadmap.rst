.. _sec_roadmap:

Road Map
========

Overview
--------

+-----------------------------------+------------------+-------------------------------+
| Task                              | Priority         | Current State                 |
+===================================+==================+===============================+
| Write Documentation               | High             | Started with a long way to go |
+-----------------------------------+------------------+-------------------------------+
| Simplify configurations           | High             | Not Started                   |
+-----------------------------------+------------------+-------------------------------+
| Develop Data Conventions          | High             | Not Started                   |
+-----------------------------------+------------------+-------------------------------+
| Improve Blockwise Post-Processing | Low              | Not Started                   |
+-----------------------------------+------------------+-------------------------------+
| Simplify Array handling           | High             | Completed                     |
+-----------------------------------+------------------+-------------------------------+

Detailed Road Map
-----------------

 - [ ] Write Documentation
     - [ ] tutorials: not more than three, simple and continuously tested (with Github actions, small U-Net on CPU could work)
         - [x] Basic tutorial: train a U-Net on a toy dataset
           - [ ] Parametrize the basic tutorial across tasks (instance/semantic segmentation).
           - [ ] Improve visualizations. Move some simple plotting functions to DaCapo.
           - [ ] Add a pure pytorch implementation to show benefits side-by-side
           - [ ] Track performance metrics (e.g., loss, accuracy, etc.) so we can make sure we aren't regressing
         - [ ] semantic segmentation (LM and EM)
         - [ ] instance segmentation (LM or EM, can be simulated)
     - [ ] general documentation of CLI, also API for developers (curate docstrings)
 - [ ] Simplify configurations
     - [ ] Depricate old configs
     - [ ] Add simplified config for simple cases
     - [ ] can still get rid of `*Config` classes
 - [ ] Develop Data Conventions
     - [ ] document conventions
     - [ ] convenience scripts to convert dataset into our convention (even starting from directories of PNG files)
 - [ ] Improve Blockwise Post-Processing
     - [ ] De-duplicate code between “in-memory” and “block-wise” processing
         - [ ] have only block-wise algorithms, use those also for “in-memory”
         - [ ] no more “in-memory”, this is just a run with a different Compute Context
     - [ ] Incorporate `volara` into DaCapo (embargo until January)
     - [ ] Improve debugging support (logging of chain of commands for reproducible runs)
     - [ ] Split long post-processing steps into several smaller ones for composability (e.g., support running each step independently if we want to support choosing between `waterz` and `mutex_watershed` for fragment generation or agglomeration)
 - [x] Incorporate `funlib.persistence` adaptors.
     - [x] all of those can be adapters:
         - [x] Binarize Labels into Mask
         - [x] Scale/Shift intensities
         - [ ] Up/Down sample (if easily possible)
         - [ ] DVID source
         - [x] Datatype conversions
         - [x] everything else
     - [ ] simplify array configs accordingly

Can Have
--------

 - [ ] Support other stats stores. Too much time, effort and code was put into the stats and didn’t provide a very nice interface:
     - [ ] defining variables to store
     - [ ] efficiently batch writing, storing and reading stats to both files and mongodb
     - [ ] visualizing stats.
     - [ ] Jeff and Marwan suggest MLFlow instead of WandB
 - [ ] Support for slurm clusters
 - [ ] Support for cloud computing (AWS)
 - [ ] Lazy loading of dependencies (import takes too long)
 - [ ] Support bioimage model spec for model dissemination

Non-Goals (for v1.0)
--------------------

- custom dash board
- GUI to run experiments