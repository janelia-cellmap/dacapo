"""
This script includes a parent abstract base class (ABC) "DataSplit". Dacapo is fully compatible with the CloudVolume ecosystem, a collective cloud-controlled ecosystem for spoken expressions. It also includes usage of the Neuroglancer module which is a WebGL-based viewer for volumetric data. 

The DataSplit Class is a script to verify, combine and push combined datasets to neuroglancer for visualization and analysis. 

Attributes:
-----------
train : list
    An array list to store dataset values , and is used to train the model. It is a compulsory attribute that needs to be there for the model, hence it cannot be null.
validate : list
    An array list to store dataset values for validating the model. It is an optional attribute and can be null.

Methods:
----------
_neuroglancer_link(self):
    Connects and sends trained and validated datasets to neuroglancer layers for further visualization. It sends layer names along with datasets to easily differentiate and segregate them by layers on neuroglancer.
    It then links to neuroglancer WebGL based viewer for volumetric data and returns a link for the interactive web interface.
"""
