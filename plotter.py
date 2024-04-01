import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, fixed



class Geometry:
    """
    Utility class for handling geometric properties.

    This class provides utility functions and mappings for handling geometric properties,
    such as planes, axes, and their corresponding mappings.

    Attributes:
        _planes (set[str]): Set containing plane names ('axial', 'sagittal', 'coronal').
        _plane_to_axis (dict[str, str]): Mapping of plane names to corresponding axis names.
        _axis_to_plane (dict[str, str]): Mapping of axis names to corresponding plane names.
        _plane_to_num (dict[str, int]): Mapping of plane names to numerical identifiers.
        _num_to_plane (dict[int, str]): Mapping of numerical identifiers to plane names.
        _axes_to_num (dict[str, int]): Mapping of axis names to numerical identifiers.
        _num_to_axes (dict[int, str]): Mapping of numerical identifiers to axis names.

    """

    _planes: set[str] = {'axial', 'sagittal', 'coronal'}
    
    _plane_to_axis: dict[str, str] = {'axial':'z', 'sagittal':'y', 'coronal':'x'}
    _axis_to_plane: dict[str, str] = {'z':'axial', 'y':'sagittal', 'x':'coronal'}
    
    _plane_to_num: dict[str, int] = {'axial':3, 'sagittal':1, 'coronal':2}
    _num_to_plane: dict[int, str] = {3:'axial', 1:'sagittal', 2:'coronal'}

    _axes_to_num: dict[str, int] = {'z':3, 'y':1, 'x':2}
    _num_to_axes: dict[int, str] = {2:'x', 1:'y', 3:'z'} # Number is based on numpy arrays of the nifti file

    @staticmethod
    def _get_planes() -> str:
        """
        Get a string containing all available plane names.

        Returns:
            str: A space-separated string containing all available plane names.
        """
        string: str = " ".join(Geometry._planes)
        return string




class Plotter:
    """
    Utility class for plotting 3D slices of NIfTI images interactively.    
    The plotting functions in this class are designed to visualize 3D medical
    image data stored in NIfTI format.

    Attributes:
        _plot_function_of_plane (dict): A dictionary mapping plane names 
        to their respective plotting functions.
    """

    @staticmethod
    def plot_y(img: np.ndarray[np.float64, np.float64, np.float64], 
               slice: int) -> None:
        """
        Plot a 2D slice of a 3D image along the y-axis.
        Args:
            img (numpy.ndarray): The 3D image array.
            slice (int): The slice index along the y-axis to plot.
        """
        plt.imshow(img[slice,:,:])

    @staticmethod
    def plot_x(img: np.ndarray[np.float64, np.float64, np.float64],
                slice: int) -> None:
        """
        Plot a 2D slice of a 3D image along the x-axis.
        Args:
            img (numpy.ndarray): The 3D image array.
            slice (int): The slice index along the x-axis to plot.
        """
        plt.imshow(img[:,slice,:])

    @staticmethod
    def plot_z(img: np.ndarray[np.float64, np.float64, np.float64],
                slice: int) -> None:
        """
        Plot a 2D slice of a 3D image along the z-axis.
        Args:
            img (numpy.ndarray): The 3D image array.
            slice (int): The slice index along the z-axis to plot.
        """
        plt.imshow(img[:,:,slice])

    _plot_function_of_plane = {'axial': plot_z,
                              'sagittal': plot_y, 
                              'coronal': plot_x
                              }    

    @staticmethod
    def plot_slices(img: np.ndarray[np.float64, np.float64, np.float64],
                     plane: str) -> None:
        """
        Plot interactive slices of a 3D image along a specified plane.

        Args:
            img (numpy.ndarray): The 3D image array.
            plane (str): The plane along which to plot slices ('axial', 'sagittal', or 'coronal').

        Raises:
            TypeError: If an invalid plane name is provided.
        """
        if plane not in Geometry._planes:
            raise TypeError(f"This plane does not belong to\
                             \n{Geometry._get_planes()}")

        slicer = IntSlider(min=0, 
                           max=img.shape[Geometry._plane_to_num[plane]-1]-1, 
                           step=1, 
                           description=f"{Geometry._plane_to_axis[plane]}-axis")
        
        interact(Plotter._plot_function_of_plane[plane],
                  img=fixed(img), 
                  slice=slicer)  

        
