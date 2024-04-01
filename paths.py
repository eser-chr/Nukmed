from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class Paths:
    """
    A data class representing paths used in a medical imaging processing pipeline.
    
    Attributes:
    dicom_folder: str
        Path to the folder containing DICOM files.
    nifti_folder: str
        Path to the folder where NIfTI files will be stored.
    segmentation_folder: str
        Path to the folder where segmentation results will be stored.
    results_path: str
        Path to the file where overall results will be stored.
    nifti_subfolder_name: Optional[str] = None
        Optional subfolder name within the NIfTI folder.
    non_isotropic_nifti_path: Optional[str] = None
        Path to the non-isotropic NIfTI file.
    isotropic_nifti_path: Optional[str] = None
        Path to the isotropic NIfTI file.
    lung_lr: Optional[str] = None
        Path to the lung left-right segmentation file.
    lung_ll: Optional[str] = None
        Path to the lung left-left segmentation file.
    aorta: Optional[str] = None
        Path to the aorta segmentation file.
    heart: Optional[str] = None
        Path to the heart segmentation file.
    heart_surrounding: Optional[str] = None
        Path to the heart surrounding segmentation file.
    thoracic_aorta: Optional[str] = None
        Path to the thoracic aorta segmentation file.
    thoracic_aorta_surrounding: Optional[str] = None
        Path to the thoracic aorta surrounding segmentation file.
    """

    dicom_folder: str
    nifti_folder: str
    segmentation_folder: str
    results_path: str
    nifti_subfolder_name: Optional[str] = None  
    non_isotropic_nifti_path: Optional[str] = None
    isotropic_nifti_path: Optional[str] = None
    lung_lr: Optional[str] = None
    lung_ll: Optional[str] = None
    aorta: Optional[str] = None
    heart: Optional[str] = None
    heart_surrounding: Optional[str] = None
    thoracic_aorta: Optional[str] = None
    thoracic_aorta_surrounding: Optional[str] = None

    def __str__(self):
        string_representation = "----PATHS-----\n"
        string_representation += f"Dicom folder: {self.dicom_folder}\n"
        string_representation += f"Nifti folder: {self.nifti_folder}\n"
        string_representation += f"Segmentation folder: {self.segmentation_folder}\n"
        string_representation += f"Results path: {self.results_path}\n"
        string_representation += f"Nifti subfolder name: {self.nifti_subfolder_name}\n"
        string_representation += f"Non-isotropic nifti path: {self.non_isotropic_nifti_path}\n"
        string_representation += f"Isotropic nifti path: {self.isotropic_nifti_path}\n"
        string_representation += f"Lung LR: {self.lung_lr}\n"
        string_representation += f"Lung LL: {self.lung_ll}\n"
        string_representation += f"Aorta: {self.aorta}\n"
        string_representation += f"Heart: {self.heart}\n"
        string_representation += f"Heart Surrounding: {self.heart_surrounding}\n"
        string_representation += f"Thoracic Aorta: {self.thoracic_aorta}\n"
        string_representation += f"Thoracic Aorta Surrounding: {self.thoracic_aorta_surrounding}\n"
        return string_representation

    def _mandatory_path(self)->None:
        mandatory_paths = [self.dicom_folder, self.nifti_folder, 
                        self.segmentation_folder]
        for path in mandatory_paths:
            try:
                os.makedirs(path, exist_ok=True)
            except:
                raise TypeError(f"Something went worng with initialization of {path}")
            
    
    
    def _create_path_of_nifti_subfolder(self)->None:
        unique_temp_subfolder_number = 1
        template = "temp"
        unique_foldername = f"{template}_{str(unique_temp_subfolder_number)}"

        while unique_foldername in os.listdir(self.nifti_folder):
            unique_temp_subfolder_number+= 1
            unique_foldername = f"{template}_{str(unique_temp_subfolder_number)}"
        self.nifti_subfolder_name = unique_foldername
    
    
    
    def _nifti_subfolder(self)->None:
        if self.nifti_subfolder_name == None:
            self._create_path_of_nifti_subfolder()
        self.nifti_subfolder_name = os.path.join(self.nifti_folder,self.nifti_subfolder_name)
        # print(total)
        os.makedirs(self.nifti_subfolder_name, exist_ok=True)
    


    def _to_absolute_paths(self) -> None:
        """
        Transform all paths to absolute paths.

        Returns:
            Paths: A new Paths object with absolute paths.
        """        
        self.dicom_folder=os.path.abspath(self.dicom_folder)
        self.nifti_folder=os.path.abspath(self.nifti_folder)
        self.segmentation_folder=os.path.abspath(self.segmentation_folder)
        self.results_path=os.path.abspath(self.results_path)
        self.nifti_subfolder_name=os.path.abspath(self.nifti_subfolder_name)
        self.non_isotropic_nifti_path = os.path.abspath(
            os.path.join(self.nifti_folder, 'non_isotropic.nii')
        )

        self.isotropic_nifti_path = os.path.abspath(
            os.path.join(self.nifti_folder, 'isotropic.nii')
        )
        
    

    def create_missing_directories(self) -> None:
        """
        Create missing directories if they do not exist.
        """
        self._mandatory_path()
        self._nifti_subfolder()
        self._to_absolute_paths()
