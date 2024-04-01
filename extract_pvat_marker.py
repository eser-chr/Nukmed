#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import subprocess
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

import pydicom
import nibabel as nib
import nibabel.processing
import re
from paths import Paths
from utils import filter_isolated_islands, volume_around_mask, get_array_from_nifti
from typing import Dict


def create_isotropic_nifti(paths:Paths)->None: # Heavy function (~1 min)
    '''
    Creates an isotropic image based on an anisotropic image. Both input and output are 
    nifti files.
    '''
    img = nib.load(paths.non_isotropic_nifti_path)
    pixdim = img.header.get("pixdim")[1:4]
    isotropic_target_spacing = np.min(pixdim)
    target_pixdim = [isotropic_target_spacing, ] * 3
    # the following is the bottle neck.
    resampled_img = nib.processing.resample_to_output(img, voxel_sizes=target_pixdim)   
    nib.save(resampled_img, paths.isotropic_nifti_path)



def calculate_fat_volume_in_ml(vol_for_thresholding:np.ndarray, 
                         value_range:tuple[int, int], 
                         isotropic_target_spacing, 
                         min_connected_voxels,
                         paths:Paths, 
                         structure_img:nib.Nifti1Image):

    # Calculate volume of non-thresholded VOI
    non_thres_vol = np.sum(vol_for_thresholding != 0) * (isotropic_target_spacing ** 3)
    non_thres_vol_cm3 = non_thres_vol / 1000

    # Threshold HUs
    vol_for_thresholding[(vol_for_thresholding < value_range[0]) | (vol_for_thresholding > value_range[1])] = 0
    vol_for_thresholding[(vol_for_thresholding >= value_range[0]) & (vol_for_thresholding <= value_range[1])] = 1

    # Remove isolated voxels
    if min_connected_voxels > 1:
        filter_isolated_islands(vol_for_thresholding, min_voxels=min_connected_voxels)

    thres_nii = nib.Nifti1Image(vol_for_thresholding, structure_img.affine, structure_img.header)
    nib.save(thres_nii, os.path.join(paths.segmentation_folder, "surrounding_thresholded.nii.gz"))

    thres_vol = np.sum(vol_for_thresholding) * (isotropic_target_spacing ** 3)

    thres_vol_cm3 = thres_vol / 1000
    relative_vol = thres_vol_cm3 / (thres_vol_cm3 + non_thres_vol_cm3)

    return thres_vol_cm3, relative_vol




def get_lastZ_of_all(masks):
    lista = np.array([np.where(np.any(obj, axis=(0, 1)))[0][0] for obj in masks])
    return np.max(lista)


def convert_dicom_to_nifti(paths)->None:
    cmd_str = f"dcm2niix -o {paths.nifti_subfolder_name} {paths.dicom_folder}"
    subprocess.run(cmd_str, shell=True)


def total_segmentator(paths, anatomical_structure, radiomics):
    # Another bottleneck (~5 min.)
    cmd_str = f"TotalSegmentator -i {paths.isotropic_nifti_path} -o {paths.segmentation_folder} --roi_subset " \
                f"{anatomical_structure} lung_lower_lobe_left lung_lower_lobe_right heart --body_seg"

    if radiomics:
        cmd_str = cmd_str + " --radiomics"
    subprocess.run(cmd_str, shell=True)


def move_nifti_to_nifti_folder(paths)->None:
    pattern = re.compile(r'^((?!ROI1\.nii).)*\.nii$')
    nifti_file = [file for file 
                  in os.listdir(paths.nifti_subfolder_name)
                  if pattern.match(file)][0]
    shutil.move(os.path.join(paths.nifti_subfolder_name, nifti_file),
                 paths.non_isotropic_nifti_path)



class Nifti_Options:
    def __init__(self):
        self.affine = None
        self.header = None
        self.pixdim = None
        self.isotropic_target_spacing = None
        self.target_pixdim = None

    def load(self, nifti_img):
        self.affine = nifti_img.affine
        self.header = nifti_img.header
        self.pixdim = self.header.get("pixdim")[1:4]
        self.isotropic_target_spacing = np.min(self.pixdim)
        self.target_pixdim = [self.isotropic_target_spacing, ]*3


class Organ:
    def __init__(self, name:str, path:str):
        self.name = name
        self.path = path
        self.img = None
        self.mask = None
        self.nifti_options = Nifti_Options()
    
    def load_image(self):
        self.img = nib.load(self.path)

    def load_array(self):
        self.mask = get_array_from_nifti(self.path)
    
    def load_nifti_options(self):
        if self.img == None:
            self.load_image()
        self.nifti_options.load(self.img)


    def get_image(self):
        if self.img ==None:
            self.load_image()
        return self.img
    
    def get_mask(self):
        if self.mask==None:
            self.load_array()
        return self.mask
    




def create_obj_organ_names(paths, organs_to_names):
    res = {}
    for organ in organs_to_names.keys():
        res[organ] = Organ(organ,
                           os.path.join(paths.segmentation_folder, 
                                        organs_to_names[organ])
                        )
    return res


def split_organ_z(z_value:int, organ_mask:np.ndarray):

    """ 
    Returns the part of the organ that's above a value on z-axis.
    """
    new = organ_mask.copy()
    new[:, :, z_value:] = 1
    new[:, :, :z_value] = 0
    return new * organ_mask


def save_nifti(mask:np.ndarray, organs:dict[str, Organ], path:str)->None:
        nifti = nib.Nifti1Image(mask, 
                                organs["aorta"].nifti_options.affine,
                                organs["aorta"].nifti_options.header)
        nib.save(nifti, path)



def periaortic_fat_marker(paths:Paths, min_connected_voxels=1, 
                          value_range=(-195, -45), 
                          dilation_length_mm=5,
                          radiomics=False, 
                          precomputed_mask=None):
    
 
    paths.create_missing_directories()
    convert_dicom_to_nifti(paths)
    move_nifti_to_nifti_folder(paths) 
    create_isotropic_nifti(paths)      
    total_segmentator(paths, "aorta", radiomics)

    organs_to_names = {"aorta":"aorta.nii.gz", 
                       "lung_ll":"lung_lower_lobe_left.nii.gz", 
                       "lung_lr":"lung_lower_lobe_right.nii.gz", 
                       "heart":"heart.nii.gz"}
    
    organs:Dict[str, Organ] = create_obj_organ_names(paths, organs_to_names)
    organs["isotropic"] = Organ("isotropic", paths.isotropic_nifti_path)

    for organ in organs.keys():
        organs[organ].load_image()
        organs[organ].load_array()
    
    organs["aorta"].load_nifti_options()
    organs["isotropic"].load_nifti_options()

         
    last_lung_z_index = get_lastZ_of_all([organs["lung_ll"].mask, organs["lung_lr"].mask])

    thoracic_aorta_mask = split_organ_z(last_lung_z_index, organs["aorta"].mask)
    print("Start real code")

    save_nifti(thoracic_aorta_mask, organs, os.path.join(paths.segmentation_folder, f"thoracic_aorta.nii.gz"))

    organs["thoracic_aorta"] = Organ("aorta", os.path.join(
                        paths.segmentation_folder, "thoracic_aorta.nii.gz"))
    organs["thoracic_aorta"].load_array()
    thoracic_aorta = organs["isotropic"].get_mask() * organs["thoracic_aorta"].get_mask() 
    
    
    surrounding_mask = volume_around_mask(thoracic_aorta_mask, organs["isotropic"].nifti_options.pixdim, dilation_length_mm)
    save_nifti(surrounding_mask, organs, os.path.join(paths.segmentation_folder, f"thoracic_aorta_surrounding.nii.gz"))
    thoracic_aorta_surrounding = organs["isotropic"].get_mask() * surrounding_mask

    # ------------

    heart = organs["isotropic"].get_mask() * organs["heart"].get_mask()
    surrounding_mask = volume_around_mask(organs["heart"].mask, organs["isotropic"].nifti_options.pixdim, dilation_length_mm)
    save_nifti(surrounding_mask, organs, os.path.join(paths.segmentation_folder, f"heart_surrounding.nii.gz"))
    heart_surrounding = organs["isotropic"].get_mask() * surrounding_mask

    # ------------
    # create csv-----------------------

    markers = pd.DataFrame()

    target_volume_map = {
        "Periaortic": thoracic_aorta_surrounding,
        "Pericardial": heart_surrounding,
        "Aorta": thoracic_aorta,
        "Heart": heart,
    }

    for target_area in target_volume_map.keys():

        thres_vol_cm3, relative_vol = calculate_fat_volume_in_ml(
            target_volume_map[target_area], value_range,
            organs["isotropic"].nifti_options.isotropic_target_spacing, 
                                   min_connected_voxels, 
                                   paths, 
                                   organs["aorta"].get_image()
                                   )


        markers[f"{target_area} absolute volume (ml)"] = [thres_vol_cm3]
        markers[f"{target_area} relative volume"] = [relative_vol]

    return markers



