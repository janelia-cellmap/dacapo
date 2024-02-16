```
"""
A class to create a configuration for concatenated arrays. This configuration is used 
to build a more complex array structure from a set of simpler arrays.

Attributes:
    array_type (ConcatArray): Class of the array, inherited from the ArrayConfig class. 
    channels (List[str]): An ordered list of channels in source_arrays. This order 
                          determines the resulting array's order.
    source_array_configs (Dict[str, ArrayConfig]): A dictionary mapping channels to 
                                                  their respective array config.
                                                  If a channel has no ArrayConfig, it 
                                                  will be filled with zeros.
    default_config (Optional[ArrayConfig]): Defines a default array configuration for 
                                            channels. Only needed if some channels' 
                                            configurations are not provided. If not 
                                            provided, missing channels will be filled 
                                            with zeros.

"""
```