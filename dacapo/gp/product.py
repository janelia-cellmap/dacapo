"""
This script defines a Python class 'Product' in the gunpowder library which multiplies two arrays.

Attributes:
    x1_key (gp.ArrayKey): The ArrayKey for the first array.
    x2_key (gp.ArrayKey): The ArrayKey for the second array.
    y_key (gp.ArrayKey): The ArrayKey for the resulting array after multiplication.

Methods:
    __init__(self, x1_key: gp.ArrayKey, x2_key: gp.ArrayKey, y_key: gp.ArrayKey): 
        Initializes the Product class with x1_key, x2_key, and y_key attributes.
    
    setup(self):
        Configures the batch filter that allows skipping of the node in the pipeline if data isn't available or not requested.
        Provides y_key array derived from the duplicate of x1_key specification.

    prepare(self, request):
        Accepts batch request, returns dependencies including the requests of array x1_key and array x2_key.

    process(self, batch, request):
        Accepts batch and request data, processes and returns outputs batch containing y_key array, 
        which is the product of x1_key and x2_key arrays data.
"""
