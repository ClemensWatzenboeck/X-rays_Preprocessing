import logging
import os
import sys
import io
from datetime import datetime
from typing import Dict
 
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
 
# Reconfigure stdout/stderr to replace characters that can't be encoded
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

###### Step 1

def setup_logger(output_folder):

    """Sets up a logger that writes to both the console and a file."""
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate a unique log filename based on the current time
    log_filename = datetime.now().strftime("pipeline_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(output_folder, log_filename)

    #Create a custom logger
    logger = logging.getLogger("Preprocessing")
    logger.setLevel(logging.DEBUG)

    #Create handlers
    c_handler = logging.StreamHandler()  
    f_handler = logging.FileHandler(log_path)
    c_handler.setLevel(logging.INFO)       
    f_handler.setLevel(logging.DEBUG)     

    #Create formatters and add them to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    #Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def extract_metadata(file_path: str, tag_map: Dict[str, str]) -> Dict:
    """Extracts DICOM tags for a single file."""
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        
        # Build dictionary for this specific file
        entry = {}
        for label, tag in tag_map.items():
            if reader.HasMetaDataKey(tag):
                val = reader.GetMetaData(tag).strip()
                entry[label] = val if val != "" else "emptystr"
            else:
                entry[label] = np.nan
        
        # Add file info
        entry['filename_old'] = os.path.basename(file_path)
        entry['filepathname_old'] = file_path
        return entry
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
def get_unique_metadata(df, exclude_cols):
    """
    Extracts unique non-null values for specified columns and 
    returns a summary DataFrame.
    """
    
    # Filter columns we want to examine
    cols_to_process = [c for c in df.columns if c not in exclude_cols]
    
    # Create a dictionary of unique, non-null values for each column
    # .dropna().unique() is significantly faster than list(dict.fromkeys(...))
    unique_map = {
        col: pd.Series(df[col].dropna().unique()) 
        for col in cols_to_process
    }
    
    return pd.DataFrame(unique_map)

###### Step 2

def categorize_column(df, target_col, primary_tag, mapping_dict, fallbacks=None):
    """
    Standardized logic: 
    1. Check primary DICOM tag against a list.
    2. If not found, check fallback columns (Series/Study Description).
    """
    # Initialize with NaN
    df[target_col] = np.nan
    
    # Priority 1: Direct Mapping from Primary Tag
    for label, keywords in mapping_dict.items():
        mask = df[primary_tag].isin(keywords)
        df.loc[mask, target_col] = label

    # Priority 2: Fallback (only for rows still NaN)
    if fallbacks:
        for col, fallback_mapping in fallbacks:
            if not df[target_col].isna().any():
                break
            for label, keywords in fallback_mapping.items():
                mask = df[target_col].isna() & df[col].isin(keywords)
                df.loc[mask, target_col] = label
           
    return df

# Function for checking if both left & right hands are present on the same date
def get_pair_status(row, pair_lookup):
    # Only relevant for Hand DP
    if row['bodypart_new'] != 'H' or row['view_position_new'] != 'dp':
        return "N/A"
    
    key = (row['pat_id'], row['study_date'])
    available_sides = pair_lookup.get(key, set())
    
    current = row['laterality_new']
    
    if current == 'B':
        return "1"

    other = 'R' if current == 'L' else 'L'
    
    return "1" if other in available_sides else f"Missing_{other}_Side"

    

###### Step 3
 
def split_dicom(src_path, row, side):
    """
    Reads a DICOM, crops it to the Left or Right half, 
    updates metadata, and saves to the destination.
    """
    
    # Work from a local copy so the caller's row is never mutated (Bug 4 fix)
    row = row.copy()
    ds = pydicom.dcmread(src_path)

    # fix invertion
    if row['photometric_interpretation_new'] == 'MOne':
        ds = invert_monochrome(ds)
        row['photometric_interpretation_new'] = 'MTwo'        

    pixel_data = ds.pixel_array
    _, width = pixel_data.shape
    mid = width // 2

    cropped_pixels = pixel_data[:, mid:] if side == 'R' else pixel_data[:, :mid]
        
    # Update DICOM header metadata
    ds.Rows, ds.Columns = cropped_pixels.shape
    ds.PixelData = cropped_pixels.tobytes()
    #ds.Laterality = side

    # fix side
    if side == 'R':
        ds = mirror_right_to_left(ds)
    
    new_filename = (
        f"{row['pat_id']}_{row['study_date']}_{row['bodypart_new']}_"
        f"{side}_{row['view_position_new']}_{row['photometric_interpretation_new']}_"
        f"{row['dup_suffix']}_split"
    )
    
    return ds, new_filename

def check_dicom_metadata(ds, row):
    """
    Standardizes tags for the training dataset.
    """
    ds.BodyPartExamined = row["bodypart_new"]
    ds.ViewPosition = row["view_position_new"]
    ds.Laterality = row["laterality_new"]
    
    # Store the filename inside the DICOM for easy viewing later
    ds.ReferringPhysicianName = row['filename_new_dupl']

    return ds

def invert_monochrome(ds):
    """
    Inverts the pixel array of a DICOM object to convert MONOCHROME1 to MONOCHROME2.
    Does NOT save the file; returns the modified dataset object.
    """
    
    pixel_array = ds.pixel_array
    max_val = (2 ** ds.BitsStored) - 1
    inverted_pixels = max_val - pixel_array
    ds.PixelData = inverted_pixels.tobytes()
    ds.PhotometricInterpretation = "MONOCHROME2"
    return ds

def mirror_right_to_left(ds):
    """
    Horizontally flips a DICOM image and updates metadata.
    Use this for all Right hands (Train, Val, and Test) so they 
    anatomically match the orientation of Left hands.
    """

    mirrored_pixels = np.flip(ds.pixel_array, axis=1)
    ds.PixelData = mirrored_pixels.tobytes()
    
    return ds

 
def rebuild_filename(row):

    return (
        f"{row['pat_id']}_{row['study_date']}_{row['bodypart_new']}_"
        f"{row['laterality_new']}_{row['view_position_new']}_"
        f"{row['photometric_interpretation_new']}_{row['dup_suffix']}"
    )