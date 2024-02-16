"""
This module facilitates the conversion of various configs, objects, and paths
for the dacapo library. The usage of register hooks allows the conversion
of these classes and types to be modifiable at runtime.

Functions:
----------
    register_hierarchy_hooks(converter): register type hierarchies for conversion.
    
    register_hooks(converter): register all conversion hooks with the given converter.
    
    cls_fun(typ): convert a type string into the corresponding class. 

"""