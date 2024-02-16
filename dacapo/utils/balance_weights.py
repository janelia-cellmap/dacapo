"""
This script defined a function 'balance_weights' used in funkelab dacapo python library.
This function is used to balance the class weights in the data labels, particularly useful 
when dealing with imbalanced dataset in machine learning tasks. 

Args:
    label_data (np.ndarray): The input data labels.
    num_classes (int): Number of unique classes in the labels.
    masks (List[np.ndarray], optional): Optional list of masks to apply on labels. Defaults to empty list.
    slab: Slices to break up the array into smaller pieces.
    clipmin (float, optional): Minimum fraction to clip to when balancing weights. Defaults to 0.05.
    clipmax (float, optional): Maximum fraction to clip to when balancing weights. Defaults to 0.95.
    moving_counts(Optional[List[Dict[int, Tuple[int, int]]]]): 
    Moving counts of samples paired with their respective class. Defaults to None.

Returns:
    error_scale (np.ndarray): The balanced weights for the classes.
    moving_counts (list): Updated moving counts for further iterations.

Raises:
    AssertionError: If there are unique labels more than the expected number of classes.
    AssertionError: If labels are not in the expected range [0, num_classes).
"""