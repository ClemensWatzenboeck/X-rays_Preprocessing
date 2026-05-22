

import dicom_extremities_preprocessor as pp
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import importlib.resources
import yaml

# # Load mappings.yaml resource from pp package
# with importlib.resources.open_text("dicom_extremities_preprocessor.resources", "mappings.yaml") as f:
#     mappings = yaml.safe_load(f)
# cfg = mappings["categories"]


with importlib.resources.open_text("dicom_extremities_preprocessor.resources", "mappings_regex.yaml", encoding="utf-8") as f:
    mappings = yaml.safe_load(f)
cfg = mappings["categories"]


def add_InfosViaRegEx(step2_df, cfg=cfg):
    """
    Extract: 
      BodyPartExaminedNew, 
      LateralityNew, 
      ViewPositionNew, 
      PhotometricInterpretationNew
     from header parts with fallbacks to series/study description
    """

    # --- Bodypart ---
    step2_df = pp.utils.categorize_column_regex(
        step2_df, 'BodyPartExaminedNew', 'BodyPartExamined', cfg['bodypart'],
        fallbacks=[('StudyDescription',   cfg['study_description_bodypart']),
                   ('SeriesDescription', cfg['series_description_bodypart'])]
        )
    step2_df['BodyPartExaminedNew'] = step2_df['BodyPartExaminedNew'].map({'foot': 'F', 'hand': 'H', 'other': 'O'})

    


    # --- Laterality ---
    step2_df = pp.utils.categorize_column_regex(step2_df, 'LateralityNew', 'Laterality', 
                                cfg['laterality'], 
                                fallbacks =[('ViewPosition',       cfg['view_position_laterality']),
                                            ('SeriesDescription',  cfg['series_description_laterality'])])

    # # --- View Position ---
    step2_df = pp.utils.categorize_column_regex(step2_df, 'ViewPositionNew', 'ViewPosition', 
                                cfg['view_position_viewposition'], 
                                fallbacks = [('SeriesDescription', cfg['series_description_viewposition'])])


    # # --- Photometric ---
    step2_df = pp.utils.categorize_column_regex(step2_df, 'PhotometricInterpretationNew', 'PhotometricInterpretation', 
                                cfg['photometric'])

    step2_df['PhotometricInterpretationNew'] = step2_df['PhotometricInterpretationNew'].map({'MOne': 'MOne', 'MTwo': 'MTwo', 'MOther': 'MOther'})





##### OLD CODE I dont want to delete yet

# def check_Description_for_completness(df, cfg, 
#                                             parts= ["series_description_bodypart", 
#                  #"series_description_viewposition_v2",  
#                  #"series_description_laterality", 
#                  #"series_description_bodypart"
#                  ],
#                  description_column = "SeriesDescription"):

#     not_matched_descriptions = {}
    
#     for part in parts:
#         SeriesDescriptionSet = set(df[description_column])
#         for k in cfg[part].keys():
#             SeriesDescriptionSet -= set(cfg[part][k])
#         not_matched_descriptions[part] = SeriesDescriptionSet
    
#     return not_matched_descriptions



# def add_BodyPartExaminedNew_LateralityNew_ViewPositionNew_PhotometricInterpretationNew(step2_df, cfg):
#     """
#     Extract: 
#       BodyPartExaminedNew, 
#       LateralityNew, 
#       ViewPositionNew, 
#       PhotometricInterpretationNew
#      from header parts with fallbacks to series/study description
#     """

#     # --- Bodypart ---
#     step2_df = pp.utils.categorize_column(
#         step2_df, 'BodyPartExaminedNew', 'BodyPartExamined', cfg['bodypart'],
#         fallbacks=[('StudyDescription',   cfg['study_description_bodypart']),
#                    ('SeriesDescription', cfg['series_description_bodypart'])]
#         )
    #step2_df['BodyPartExaminedNew'] = step2_df['BodyPartExaminedNew'].map({'foot': 'F', 'hand': 'H', 'other': 'O'})

    


    # # --- Laterality ---
    # step2_df = pp.utils.categorize_column(step2_df, 'LateralityNew', 'Laterality', 
    #                             cfg['laterality'], 
    #                             fallbacks =[('ViewPosition',       cfg['view_position_laterality']),
    #                                         ('SeriesDescription',  cfg['series_description_laterality'])])

    # # # --- View Position ---
    # step2_df = pp.utils.categorize_column(step2_df, 'ViewPositionNew', 'ViewPosition', 
    #                             cfg['view_position_viewposition'], 
    #                             fallbacks = [('SeriesDescription', cfg['series_description_viewposition_v2'])])


    # # # --- Photometric ---
    # step2_df = pp.utils.categorize_column(step2_df, 'PhotometricInterpretationNew', 'PhotometricInterpretation', 
    #                             cfg['photometric'])

    # step2_df['PhotometricInterpretationNew'] = step2_df['PhotometricInterpretationNew'].map({'MOne': 'MOne', 'MTwo': 'MTwo', 'MOther': 'MOther'})
