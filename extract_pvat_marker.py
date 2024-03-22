#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 28 16:57 2023

Extraction of periaortic markers

Current markers include:
- Absolute perioatric fat volume
- Relative periaortic fat volume

@author: cspielvogel
"""

import os
import pathlib
import subprocess
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

import pydicom
# from nipype.interfaces.dcm2nii import Dcm2niix, Dcm2nii
import nibabel as nib
import nibabel.processing


# Depreciated in favor of os.makedirs(path, exist_ok = True)

# def create_dir_if_not_exists(path):   
#     """Create a directory with the provided path; nested paths allowed"""
#     if not os.path.exists(path):
#         pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def filter_isolated_voxels(array, min_voxels=3):
    """
    Finds islands of size bigger than min_voxels. Returns the 
    Return array with isolated cells removed
    :param array: np.ndarray to be filtered. Needs to be an integer.
    :return: np.ndarray without clusters containing less than minimum_voxels connected voxels
    """

    filtered_array = np.copy(array)

    # Include diagonals for connectivity check
    struct = np.ones((3,) * len(array.shape))

    # Label each cluster of values with an integer
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)

    # Get size of each cluster
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))

    # Get mask for small regions
    area_mask = (id_sizes < min_voxels)

    # Remove small regions from array
    filtered_array[area_mask[id_regions]] = 0

    return filtered_array


def fuse_masks(list_of_masks):
    """
    Fuse binary masks for heart subcomponents such as in total segmentator

    :param list_of_masks: iterable (e.g. list) containing the numpy arrays with binary masks to be combined
    :return: np.ndarray containing the combined binary mask
    """

    if len(list_of_masks) == 0:
        print("[Error] List of masks for creation of combined mask is empty")

    fused_mask = np.zeros(list_of_masks[0].shape)
    for mask in list_of_masks:
        array_mask = mask.get_fdata()

        fused_mask += array_mask

    # Set voxels with overlapping masks to 1
    fused_mask = (fused_mask > 0) * 1

    # Convert back to nifti
    fused_mask = nib.Nifti1Image(fused_mask.astype(np.int16), affine=list_of_masks[0].affine)

    return fused_mask


def volume_around_mask(voi, pixdim, dilation_length_mm=5):
    """
    Get mask surrounding another mask by a given length
    :param voi: np.ndarray binary; volume to be surrounded
    :param pixdim: tuple containing isometric voxel dimensions
    :param dilation_length_mm: int indicating the number of millimeters for dilation along each axis
    :return: mask of VOI surrounding
    """

    # Get number of voxels equivalent to given dilation length
    num_vox = np.round(dilation_length_mm / pixdim[0])

    # Dilate mask
    dilated_mask = ndimage.binary_dilation(voi, iterations=int(num_vox)) * 1

    # Subtract original from dilated mask to retrieve surrounding volume
    surrounding_mask = dilated_mask - voi

    return surrounding_mask


def periaortic_fat_marker(dcm_path, nii_path=".", segmentation_path=".", anatomical_structure="aorta",
                          min_connected_voxels=1, value_range=(-195, -45), dilation_length_mm=5,
                          radiomics=False, precomputed_mask=None):
    """
    Compute a thresholded volume within or around a given anatomical structure (intended use: CT).
    All intermediate files including the original, isotropic resampled, anatomical structure segmentation mask,
    thresholded VOI mask and radiomic features are saved in the intermediate file paths provided.

    :param dcm_path: str indicating the directory path in which the input dicom files are stored
    :param nii_path: str indicating the directory path in which the intermediate nifti files are stored
    :param segmentation_path: str indicating the directory path in which the intermediate nifti segmentation files are
                              stored
    :param anatomical_structure: str indicating an anatomical structure as specified in
                                 github.com/wasserth/TotalSegmentator
    :param min_connected_voxels: int; any voxels in connection with less other voxels will be removed in the output
                                 (diagonals are considered connections)
    :param value_range: tuple containing two integers indicating the range to be thresholded in the format (lower_bound,
                        upper_bound), only used if surrounding is set to True
    :param dilation_length_mm: float or int indicating the length (in mm) for dilation around the anatomical structure
                               along each spatial axis, only used if surrounding is set to True
    :param radiomics: bool indicating whether to also extract radiomic features from the anatomical structure as json
    :param precomputed_mask: np.ndarray 3D containing a binary mask of an anatomical reference region (must have the same
                             dimensions as the resampled dicom array
    :return: tuple containing the absolute thresholded area and the volume of the non-thresholded volume in cm^3
    """
    # CREATE FOLDERS -------------------------------------------------------------------
    # Create intermediate directories if not existing

    os.makedirs(nii_path, exist_ok=True)
    os.makedirs(segmentation_path, exist_ok=True)

    # Create a temporary unique ID for output nifti folder
    unique_foldername = 1
    while str(unique_foldername) in os.listdir(nii_path):
        unique_foldername += 1
    unique_foldername = str(unique_foldername)

    os.makedirs(os.path.join(nii_path, unique_foldername), exist_ok=True)
    #------------------------------------------------------------------------------------




    # CONVERT DICOM TO NIFTI and deal with files
    # Its ~15x faster in the terminal that way, rather than using dicom package inside python!!!

    cmd_str = f"dcm2niix -o {os.path.join(nii_path, unique_foldername)} {dcm_path}"
    subprocess.run(cmd_str, shell=True)


    # Rename nifti file and move to nifti folder
    for file in os.listdir(os.path.join(nii_path, unique_foldername)):
        print(file)
        if file.endswith(".nii") and not file.endswith("ROI1.nii"):
            nifti_file = file
    nifti_file_path = os.path.join(os.path.join(nii_path, unique_foldername), nifti_file)
    target_file_path = os.path.join(nii_path, unique_foldername) + ".nii"
    shutil.move(nifti_file_path, target_file_path)
    # shutil.rmtree(os.path.join(nii_path, unique_foldername)) to remove temp files
    # ------------------------------------------------------------------------


    # CREATE ISOTROPIC NIFTI FILE---------------------
    img = nib.load(target_file_path)
    pixdim = img.header.get("pixdim")[1:4]
    isotropic_target_spacing = np.min(pixdim)
    target_pixdim = [isotropic_target_spacing, ] * 3

    # Resample image to target voxel spacing
    resampled_img = nib.processing.resample_to_output(img, voxel_sizes=target_pixdim)

    # Save resampled nifti image
    resampled_path = target_file_path.rstrip(".nii.gz") + "-isotropic.nii.gz"
    nib.save(resampled_img, resampled_path)
    #--------------------------------------------------

    

    # Create scan-specific segmentation result directory
    segmentation_path_scanwise = os.path.join(segmentation_path, target_file_path.rstrip(".nii"))
    os.makedirs(segmentation_path_scanwise, exist_ok=True)

    # Set precomputed mask as anatomical reference structure segmentation
    if precomputed_mask is not None:
        structure_mask = precomputed_mask

    # If no precomputed mask set, segment anatomical structure
    else:
        cmd_str = f"TotalSegmentator -i {resampled_path} -o {segmentation_path_scanwise} --roi_subset " \
                  f"{anatomical_structure} lung_lower_lobe_left lung_lower_lobe_right heart --body_seg"

        if radiomics:
            cmd_str = cmd_str + " --radiomics"

        subprocess.run(cmd_str, shell=True)

        # Get segmentation nifti file path
        structure_seg_path = os.path.join(segmentation_path_scanwise, f"{anatomical_structure}.nii.gz")

        # Load anatomical structure mask
        img = nib.load(structure_seg_path)
        structure_mask = img.get_fdata()

    # Remove transversal slices of the aorta below inferior margin of the lung
    lung_lowerleft_seg_path = os.path.join(segmentation_path_scanwise, "lung_lower_lobe_left.nii.gz")
    lung_lowerright_seg_path = os.path.join(segmentation_path_scanwise, "lung_lower_lobe_right.nii.gz")
    lung_ll = nib.load(lung_lowerleft_seg_path)
    lung_lr = nib.load(lung_lowerright_seg_path)
    lung_ll = lung_ll.get_fdata()
    lung_lr = lung_lr.get_fdata()
    last_slice_index_lll = np.where(np.any(lung_ll, axis=(0, 1)))[0][0]
    last_slice_index_llr = np.where(np.any(lung_lr, axis=(0, 1)))[0][0]
    last_lung_slice_index = np.max((last_slice_index_lll, last_slice_index_llr))

    thoracic_area_mask = structure_mask.copy()
    thoracic_area_mask[:, :, last_lung_slice_index:] = 1
    thoracic_area_mask[:, :, :last_lung_slice_index] = 0
    thoracic_aorta_mask = thoracic_area_mask * structure_mask

    lung_thres_nii = nib.Nifti1Image(thoracic_aorta_mask, img.affine, img.header)
    nib.save(lung_thres_nii, os.path.join(segmentation_path_scanwise, "thoracic_aorta.nii.gz"))

    # Get VOI containing the anatomical structure segmented
    thoracic_aorta = resampled_img.get_fdata() * thoracic_aorta_mask

    # Get VOI around the aorta
    surrounding_mask = volume_around_mask(thoracic_aorta_mask, target_pixdim, dilation_length_mm)

    # Save surrounding mask
    surrounding_mask_nii = nib.Nifti1Image(surrounding_mask, img.affine, img.header)
    nib.save(surrounding_mask_nii, os.path.join(segmentation_path_scanwise, "surrounding.nii.gz"))

    # Apply surrounding mask to resampled image
    thoracic_aorta_surrounding = resampled_img.get_fdata() * surrounding_mask

    # Get heart segmentations

    #Add an if else, because in my version of total splicer heart is one piece.

    # r_atrium = os.path.join(segmentation_path_scanwise, "heart_atrium_right.nii.gz")
    # r_ventricle = os.path.join(segmentation_path_scanwise, "heart_ventricle_right.nii.gz")
    # l_atrium = os.path.join(segmentation_path_scanwise, "heart_atrium_left.nii.gz")
    # l_ventricle = os.path.join(segmentation_path_scanwise, "heart_ventricle_left.nii.gz")
    # myocardium = os.path.join(segmentation_path_scanwise, "heart_myocardium.nii.gz")

    # r_atrium = nib.load(r_atrium)
    # r_ventricle = nib.load(r_ventricle)
    # l_atrium = nib.load(l_atrium)
    # l_ventricle = nib.load(l_ventricle)
    # myocardium = nib.load(myocardium)

    # heart_mask = fuse_masks([r_atrium, r_ventricle, l_atrium, l_ventricle, myocardium]).get_fdata()
    heart_path = os.path.join(segmentation_path_scanwise, "heart.nii.gz")
    heart_mask = nib.load(heart_path)
    heart_mask = heart_mask.get_fdata()
    # Get VOI containing the anatomical structure segmented
    heart = resampled_img.get_fdata() * heart_mask

    # Get VOI around the heart
    surrounding_mask = volume_around_mask(heart_mask, target_pixdim, dilation_length_mm)

    # Save surrounding mask
    surrounding_mask_nii = nib.Nifti1Image(surrounding_mask, img.affine, img.header)
    nib.save(surrounding_mask_nii, os.path.join(segmentation_path_scanwise, "surrounding.nii.gz"))

    # Apply surrounding mask to resampled image
    heart_surrounding = resampled_img.get_fdata() * surrounding_mask

    markers = pd.DataFrame()

    target_volume_map = {
        "Periaortic": thoracic_aorta_surrounding,
        "Pericardial": heart_surrounding,
        "Aorta": thoracic_aorta,
        "Heart": heart,
    }

    for target_area in target_volume_map.keys():

        vol_for_thresholding = target_volume_map[target_area]

        # Calculate volume of non-thresholded VOI
        non_thres_vol = np.sum(vol_for_thresholding != 0) * (isotropic_target_spacing ** 3)
        non_thres_vol_cm3 = non_thres_vol / 1000

        # Threshold HUs
        vol_for_thresholding[(vol_for_thresholding < value_range[0]) | (vol_for_thresholding > value_range[1])] = 0
        vol_for_thresholding[(vol_for_thresholding >= value_range[0]) & (vol_for_thresholding <= value_range[1])] = 1

        # Remove isolated voxels
        if min_connected_voxels > 1:
            filter_isolated_voxels(vol_for_thresholding, min_voxels=min_connected_voxels)

        # Save thresholded mask
        thres_nii = nib.Nifti1Image(vol_for_thresholding, img.affine, img.header)
        nib.save(thres_nii, os.path.join(segmentation_path_scanwise, "surrounding_thresholded.nii.gz"))

        # Calculate thresholded volume by multiplying number of mask voxels with voxel volume
        thres_vol = np.sum(vol_for_thresholding) * (isotropic_target_spacing ** 3)

        # Convert cubic mm to cubic cm
        thres_vol_cm3 = thres_vol / 1000

        markers[f"{target_area} absolute volume (ml)"] = [thres_vol_cm3]
        markers[f"{target_area} relative volume"] = [thres_vol_cm3 / (thres_vol_cm3 + non_thres_vol_cm3)]

    return markers


def main():
    dcm_paths = [
        "/home/cspielvogel/DataStorage/PeriaorticFat/Data/SCHREITEL-BETTINA-MAG_-FH/RX10002534044340/AC-REST-I30S",
        "/home/cspielvogel/DataStorage/PeriaorticFat/Data/SCHREITEL-BETTINA-MAG_-FH/RX10002534044340/AC-STRESS-I30S"
    ]

    nii_path = "/home/cspielvogel/DataStorage/PeriaorticFat/Data/Nifti_lehner-subprocess"
    segmentation_path = "/home/cspielvogel/DataStorage/PeriaorticFat/Data/Segmentations_lehner-subprocess"
    result_path = "/home/cspielvogel/DataStorage/PeriaorticFat/Results/periaortic_fat_markers_lehner-subprocess.csv"

    # Initialize result table
    results = pd.DataFrame()

    # Get volume for each scan
    for dcm_path in dcm_paths:
        markers = periaortic_fat_marker(dcm_path,
                                                           nii_path=nii_path,
                                                           segmentation_path=segmentation_path,
                                                           anatomical_structure="aorta",
                                                           min_connected_voxels=3,
                                                           value_range=(-195, -45),
                                                           # Periaortic fat: Lehman SJ et al., Atherosclerosis 2010
                                                           dilation_length_mm=5,
                                                           radiomics=False)

    results = pd.concat([results, markers])
    results.index = dcm_paths
    results.to_csv(result_path, sep=";")

    print(results)


if __name__ == "__main__":
    main()