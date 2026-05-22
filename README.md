### Repository for an X-ray preprocessing pipeline.


### Install 
I moved most parts to a python package. 
You can install e.g. with: 
```bash 
pip install -e . 
```

```python
import dicom_extremities_preprocessor as pp 
```


Basically it is a hirachical keyword matching dependent on the DICOM header. 
The inital version was written by Thomas Deimel and can be seen here: 
[AutoscoRA: Thomas' keyword matching for metadata extraction](https://github.com/cirmuw/ChronoCon/blob/main/ra_utils/autoscora/autoscoRA_Preprocessing/rename_and_filter/Step2_file_df_renaming.py)


Some parts of Viktoriia are still only in `src`. Look there for the full preprocessing pipeline. 

### Description of `src` from Viktorria: 

It downloads raw DICOM files extracted from the server and performs the following operations:

- Reads DICOM metadata
- Fills missing metadata tags using information extracted from the description tag
- Filters images to retain only hand DP projections
- Splits images containing both hands if no separate left and right images are already available in the dataset
- Mirrors right-hand images to standardize orientation
- Standardizes photometric interpretation to MONOCHROME2
