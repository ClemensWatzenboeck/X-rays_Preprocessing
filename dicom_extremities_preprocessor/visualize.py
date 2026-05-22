import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom


def _load_dicom_grayscale(file_path: str) -> np.ndarray:
    ds = pydicom.dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", None) == "MONOCHROME1":
        img = img.max() - img
    return img


def _format_info_text(row: pd.Series, cols: List[str]) -> str:
    lines = []
    for col in cols:
        if col not in row.index:
            continue
        val = row[col]
        lines.append(f"{'—' if pd.isna(val) else val}")
    return "\n".join(lines)


def visualize_samples(
    df: pd.DataFrame,
    txt_info_cols: Optional[List[str]] = None,
    Nx: int = 8,
    file_path_column: str = "FilePath",
    label_below: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot DICOM images from *df* in an Nx-column grayscale grid with metadata labels.

    Parameters
    ----------
    df : DataFrame
        One row per image; typically a pre-sampled subset (e.g. ``df.sample(8 * 4)``).
    txt_info_cols : list of str
        Columns whose values are shown under (or above) each image.
    Nx : int
        Number of columns in the grid; rows are inferred from ``len(df)``.
    file_path_column : str
        Column containing paths to DICOM files.
    label_below : bool
        If True, metadata is drawn below the image; otherwise above.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib Axes
    """

    n = len(df)
    if n == 0:
        raise ValueError("df is empty")

    ncols = Nx
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.2 * ncols, 2.6 * nrows),
        squeeze=False,
    )

    for i, (_, row) in enumerate(df.iterrows()):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        path = row[file_path_column]
        try:
            img = _load_dicom_grayscale(path)
            ax.imshow(img, cmap="gray", aspect="equal")
        except Exception as exc:
            ax.imshow(np.zeros((64, 64)), cmap="gray", aspect="equal")
            ax.text(
                0.5,
                0.5,
                f"Failed to load\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
                color="red",
            )

        
        if label_below and txt_info_cols is not None:
            info = _format_info_text(row, txt_info_cols)
            ax.text(
                0.5,
                -0.08,
                info,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                wrap=True,
            )
        elif txt_info_cols is not None:
            info = _format_info_text(row, txt_info_cols)
            ax.set_title(info, fontsize=7, pad=4)
        else:
            ax.set_title("", fontsize=7, pad=4)

        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    if label_below:
        fig.subplots_adjust(bottom=0.12, hspace=0.45, wspace=0.15)
    else:
        fig.tight_layout()

    return fig, axes





# -----  Hist parts: 

def plot_histogram_of_available_combinations(dfm):

    # For "complete", each (PatientID, StudyDate) combo must have *all* four:
    # ('H','L'), ('H','R'), ('F','L'), ('F','R') for (BodyPartExaminedNew, LateralityNew) (all 'dp' view assumed from earlier mask).

    required_combos = [("H", "L"), ("H", "R"), ("F", "L"), ("F", "R")]

    # Count for each (PatientID, StudyDate) the unique (BodyPartExaminedNew, LateralityNew) pairs present
    combo_counts = (
        dfm.groupby(["PatientID", "StudyDate"])
        .apply(
            lambda x: set(zip(x["BodyPartExaminedNew"], x["LateralityNew"]))
        )
    )

    # Complete visits: those with all required_combos present
    complete_visits = combo_counts[combo_counts.apply(lambda s: all(rc in s for rc in required_combos))]

    # Cut down to those which are complete
    complete_idx = dfm.set_index(["PatientID", "StudyDate"]).index.isin(complete_visits.index)
    dfm_complete = dfm[complete_idx]

    # "Others," i.e., incomplete
    incomplete_visits = set(combo_counts.index) - set(complete_visits.index)
    dfm_incomplete = dfm[dfm.set_index(["PatientID", "StudyDate"]).index.isin(incomplete_visits)]

    # For each incomplete, get what's missing
    def missing_for_visit(pairs):
        return [rc for rc in required_combos if rc not in pairs]

    incomplete_missing = combo_counts.loc[list(incomplete_visits)].apply(missing_for_visit)

    # Example outputs:
    print(f"Number of complete visits: {len(complete_visits)}")
    print(f"Number of incomplete visits: {len(incomplete_visits)}")
    print("First few examples of missing parts per incomplete visit:")
    print(incomplete_missing.head())
    # Plot histogram: for each (PatientID, StudyDate), what *combination* of available pairs are present
    # This reduces the number of columns by grouping by what's available rather than what's missing

    # For each visit return the *present* set of (BodyPartExaminedNew, LateralityNew)
    available_counter = combo_counts.value_counts()

    # Convert each set in the index to a sorted, joined string for display
    def combo_present_str(present_set):
        # present_set is a set like {('H', 'L'), ('F', 'L')}
        present_sorted = sorted(present_set)
        return ", ".join([f"{b}{l}" for (b, l) in present_sorted])

    x_labels = [combo_present_str(present) for present in available_counter.index]
    heights = available_counter.values

    plt.figure(figsize=(max(8, len(x_labels)*0.75), 5))
    plt.bar(x_labels, heights, color='C0')
    plt.ylabel("Number of visits")
    plt.xlabel("Available (BodyPart, Lat) pairs in visit")
    plt.title("Histogram of Available (BodyPart, Laterality) Combinations per Visit")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


    ##--------------Continue with complete visits  only
    # For complete visits, count visits per patient
    visits_per_patient = complete_visits.reset_index().groupby("PatientID")["StudyDate"].nunique()



    # Median and IQR
    median_visits = np.median(visits_per_patient)
    iqr_visits = np.percentile(visits_per_patient, [25, 75])

    print(f"Median number of complete visits per patient: {median_visits}")
    print(f"IQR (25th, 75th percentile): {iqr_visits}")
    print(f"Number of patients with complete visits: {visits_per_patient.shape[0]}")
    print(f"Total number of complete visits: {len(complete_visits)}")

    # Plot histogram
    plt.figure(figsize=(8,5))
    plt.hist(visits_per_patient, bins=range(1, visits_per_patient.max()+2), edgecolor='black', align='left')
    plt.xlabel("Number of complete visits per patient")
    plt.ylabel("Number of patients")
    plt.title("Histogram of Complete Visits Per Patient")
    plt.yscale("log")
    plt.ylim(None, 2000)
    plt.xticks(range(1, visits_per_patient.max()+1))
    plt.show()

    return {"visits_per_patient": visits_per_patient}
    