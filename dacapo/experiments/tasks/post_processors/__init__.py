"""
This is the main file that loads all different post-processor classes and their parameter classes from their respective modules 
in Funkelab Dacapo Python library.

Here's an overview of the loaded classes:

1. DummyPostProcessor: Dummy Post Processor class loaded from dummy_post_processor module.
2. DummyPostProcessorParameters: Class that encapsulates parameters for Dummy Post Processor.
3. PostProcessorParameters: Base class for all Post Processor's parameters classes.
4. PostProcessor: Base class for all Post Processor classes.
5. ThresholdPostProcessor: Threshold Post Processor class loaded from threshold_post_processor module.
6. ThresholdPostProcessorParameters: Class that encapsulates parameters for Threshold Post Processor.
7. ArgmaxPostProcessor: Argmax Post Processor class loaded from argmax_post_processor module.
8. ArgmaxPostProcessorParameters: Class that encapsulates parameters for Argmax Post Processor.
9. WatershedPostProcessor: Watershed Post Processor class loaded from watershed_post_processor module.
10. WatershedPostProcessorParameters: Class that encapsulates parameters for Watershed Post Processor.

The aforementioned classes are imported using relative imports and certain warnings from linters about these imports are 
silenced with 'noqa' comments.
"""