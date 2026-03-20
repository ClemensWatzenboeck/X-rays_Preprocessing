Repository for an X-ray preprocessing pipeline.

It downloads raw DICOM files extracted from the server and performs the following operations:

- Reads DICOM metadata
- Fills missing metadata tags using information extracted from the description tag
- Filters images to retain only hand DP projections
- Splits images containing both hands if no separate left and right images are already available in the dataset
- Mirrors right-hand images to standardize orientation
- Standardizes photometric interpretation to MONOCHROME2
