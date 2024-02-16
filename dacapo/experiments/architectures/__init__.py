"""
This module publicly exposes the core components of the funkelab dacapo python library.

The module consists of major components such as ArchitectureConfig, DummyArchitectureConfig and CNNectomeUNetConfig.
Each of these come with their respective classes like Architecture, CNNectomeUNet etc.

Imports:
  - Architectures: High-level component for designing the model architecture.
  - ArchitectureConfig: High-level component for configuring the model architecture.
  - DummyArchitectureConfig, DummyArchitecture: High-level component used to create test/baseline models 
    with limited complexity for the purpose of testing or as baseline models.
  - CNNectomeUNetConfig, CNNectomeUNet: High-level components designed to create and configure CNNectomeUNet models,
    an architecture which is widely used for bio-medical applications.

Each imported component is then exposed nationally for easier access.
"""