import numpy as np
from scipy import ndimage
from typing import List
import nibabel as nib

def filter_isolated_islands(input_array: np.ndarray,
                             min_cluster_size: int = 3) -> np.ndarray:
    """
    Finds "islands" (areas of ones) of size bigger than min_cluster_size
    and creates a mask
    Returns the input array with isolated cells removed.

    Args:
        input_array (np.ndarray): Input array to be filtered. 
                                  Must be an integer array.

        min_cluster_size (int, optional): Minimum number of connected voxels 
                                        to consider as a cluster. Defaults to 3.

    Returns:
        np.ndarray: Filtered array without clusters containing 
        less than min_cluster_size connected voxels.
    """
    filtered_array = np.copy(input_array)
    neighborhood_structure = np.ones((3,) * len(filtered_array.shape))

    # Label connected regions in the array
    # labeled_array is an array with same shape as input but with an integer value
    # indicating the label of the island that this pixel belongs to
    labeled_array, num_clusters = ndimage.label(filtered_array, 
                                                structure=neighborhood_structure)

    cluster_sizes = np.array(ndimage.sum(filtered_array, labeled_array,
                                         range(num_clusters + 1)))

    small_cluster_mask = (cluster_sizes < min_cluster_size)
    filtered_array[small_cluster_mask[labeled_array]] = 0
    return filtered_array



def is_same_shape(arrays:List[np.ndarray])->bool:
    """
    Check if all arrays in a list have the same shape.
    
    Args:
        arrays (list[np.ndarray]): List of NumPy arrays.

Returns:
        bool: True if all arrays have the same shape, False otherwise.
    """
    if not arrays:
        return False  # Empty list is considered to not have the same shape

    first_shape = arrays[0].shape
    for arr in arrays[1:]:
        if arr.shape != first_shape:
            return False
    return True



def fuse_masks(list_of_masks: List[np.ndarray]) -> np.ndarray:
    """
    Fuse (spatial AND) of a list of binary masks into a single mask by element-wise multiplication.

    Args:
        list_of_masks (List[np.ndarray]): List of NumPy arrays representing binary masks.

    Returns:
        np.ndarray: Fused mask representing the intersection of all masks in the list.

    Raises:
        IndexError: If the input masks have different shapes.
    """

    if not is_same_shape(list_of_masks):
        raise IndexError("[Error] Input list of masks contains arrays with different shapes.")

    fused_mask = np.zeros(list_of_masks[0].shape)

    for mask in list_of_masks:
        fused_mask *= mask.astype(bool)

    return fused_mask



def volume_around_mask(voi: np.ndarray, pixdim: np.ndarray, 
                       dilation_length_mm: float = 5) -> np.ndarray:
    """
    Generate a volume around a binary mask by dilation.

    This function dilates the input binary mask to generate a larger volume
    around it.

    Args:
        voi (np.ndarray): Binary mask representing the volume of interest (VOI).
        pixdim (np.ndarray): Pixel dimensions (voxel size) in millimeters along each axis.
        dilation_length_mm (float, optional): Length in millimeters by which to dilate the mask. 
            Defaults to 5.

    Returns:
        np.ndarray: Binary mask representing the dilated volume around the input mask.
    """
    
    num_vox = np.round(dilation_length_mm / pixdim[0])
    dilated_mask = ndimage.binary_dilation(voi, iterations=int(num_vox)) * 1
    volume_around_mask = dilated_mask - voi

    return volume_around_mask


def get_array_from_nifti(nifti_path:str)->np.ndarray:
    obj = nib.load(nifti_path)
    return obj.get_fdata()