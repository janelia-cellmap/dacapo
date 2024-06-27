# Post processing example scripts for distribute blockwise processing of peroxisome data.

The goal of the script is to :
- Gaussian filter the data
- Threshold the distance data to get binary data 
- Apply watershed to get connected components
- Find the connected components
- Mask False Positives Mitochondria using Mitochondria data
- Merge crops
- Filter the connected components based on size
