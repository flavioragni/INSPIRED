# INSPIRED
# INSPIRED Retinal Image Preprocessing

Data preparation and exploratory analysis workflow for retinal fundus images collected in the INSPIRED project.

The repository organises retinal images by pseudonymous participant identifier, creates an image-level manifest, prepares the resolution file required by AutoMorph, and combines AutoMorph image-quality assessments with retinal vascular and optic-disc features.

> **Project status:** this repository currently contains preprocessing and exploratory notebooks rather than a complete modelling pipeline. Raw images, questionnaire data, and the AutoMorph installation are not included.

## Overview

The workflow consists of four main stages:

1. Load the INSPIRED questionnaire and identify the enrolled participants.
2. Organise the available retinal images into participant-specific directories.
3. Create image metadata and resolution files for AutoMorph.
4. Merge AutoMorph quality-control and feature-extraction outputs into a single image-level table.

```text
INSPIRED questionnaire
        │
        ├── pseudonymous participant identifiers
        │
        ▼
Raw retinal images
        │
        ├── organise images by participant
        ├── parse eye and image type
        └── create image manifest
                    │
                    ▼
        AutoMorph resolution file
                    │
                    ▼
             AutoMorph pipeline
          ┌─────────┴─────────┐
          ▼                   ▼
   Quality assessment    Retinal features
          │                   │
          └─────────┬─────────┘
                    ▼
      Combined image-level feature table
```

## Repository structure

```text
.
├── code/
│   └── data_exploration/
│       ├── explore_images.ipynb
│       └── resolution_information.csv
├── preprocessing/
│   ├── autoMorph_summary.ipynb
│   └── 20250515_autoMorph_features.csv
├── .gitignore
└── README.md
```

## Included files

### `code/data_exploration/explore_images.ipynb`

Exploratory notebook used to:

* load the INSPIRED questionnaire;
* identify the images associated with each participant;
* copy images into participant-specific directories;
* count the number of images per participant;
* inspect the image naming convention;
* distinguish colour and opacity-suppression images;
* create an image-level manifest;
* create the AutoMorph resolution file.

### `code/data_exploration/resolution_information.csv`

Resolution file prepared for AutoMorph.

It contains two columns:

```text
fundus
res
```

The bundled file contains 192 retinal images. All images are currently assigned the same resolution value:

```text
0.0039
```

The unit and value should be verified against the acquisition device and the AutoMorph configuration before processing a new dataset.

### `preprocessing/autoMorph_summary.ipynb`

Notebook used to combine:

* the original INSPIRED image manifest;
* AutoMorph module M1 image-quality predictions;
* AutoMorph module M3 macular and vascular features.

Images without a matched quality-assessment result are skipped. Images classified as bad quality, or without an available M3 feature row, are retained with missing retinal features.

### `preprocessing/20250515_autoMorph_features.csv`

Combined image-level table produced from the AutoMorph results.

The bundled snapshot contains:

```text
162 images
34 participants
78 columns
```

These columns include:

* five image metadata fields;
* one quality rating;
* 72 retinal morphometric features.

## Requirements

The notebooks require Python and Jupyter.

The main Python dependencies are:

```text
pandas
openpyxl
matplotlib
jupyter
```

Create a local virtual environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install pandas openpyxl matplotlib jupyter
```

The image-feature extraction stage also requires a separate installation of AutoMorph. AutoMorph itself is not included in this repository.

The image-organisation notebook uses the Unix `cp` command and was developed in a Linux environment.

## Data requirements

The repository does not contain the original questionnaire or retinal images.

The expected source data are:

```text
INSPIRED QUESTIONARIO.xlsx
INSPIRED IMMAGINI RETINOGRAFI/
```

The questionnaire must contain a participant identifier column named:

```text
pseudopaz
```

The notebook also displays the following questionnaire variables:

```text
età
sesso
etnia
da quanti anni ha il diabete
peso
fumatore
si definirebbe una persona fisicamente attiva?
con che frequenza pratica attività fisica?
che tipo di attività fisica pratica più frequentemente?
anemia
anemia falciforme
```

Only `pseudopaz` is used by the current image-organisation workflow.

## Image naming convention

The notebooks expect retinal-image filenames following this general structure:

```text
<PATIENT_ID>_<TIME>_<IMAGE_TYPE>_<EYE>_RetinalImage.jpg
```

Examples:

```text
46135221882988649465394_094949_Color_L_RetinalImage.jpg
75049863263406620429310_090858_Color_R_RetinalImage.jpg
```

The fields are interpreted as:

```text
PATIENT_ID  pseudonymous participant identifier
TIME        acquisition time or image-specific timestamp
IMAGE_TYPE  Color or OpacitySuppression
EYE         L or R
```

The parser relies on underscore-separated filename components. Images that do not follow this convention will be parsed incorrectly.

## Step 1: configure the paths

Open:

```text
code/data_exploration/explore_images.ipynb
```

The notebook currently contains absolute environment-specific paths:

```python
image_path = (
    "/storage/DSH/projects/inspired/data/raw_data/20250515/"
    "INSPIRED IMMAGINI RETINOGRAFI/"
)

preprocessed_path = (
    "/storage/DSH/projects/inspired/data/preprocessed/20250515/"
)

subject_list = pd.read_excel(
    "/storage/DSH/projects/inspired/data/raw_data/20250515/"
    "INSPIRED QUESTIONARIO/INSPIRED QUESTIONARIO.xlsx"
)
```

Replace these values with paths available in the local environment.

A possible directory layout is:

```text
data/
├── raw_data/
│   └── 20250515/
│       ├── INSPIRED IMMAGINI RETINOGRAFI/
│       │   ├── <retinal image>.jpg
│       │   └── ...
│       └── INSPIRED QUESTIONARIO/
│           └── INSPIRED QUESTIONARIO.xlsx
└── preprocessed/
    └── 20250515/
```

## Step 2: organise the retinal images

Start Jupyter from the repository root:

```bash
jupyter lab
```

Open and execute:

```text
code/data_exploration/explore_images.ipynb
```

For every participant listed in the questionnaire, the notebook:

1. identifies images whose filenames begin with the participant identifier;
2. creates a participant-specific output directory;
3. copies the matched images into that directory;
4. records the participant identifier, output directory, and image count.

The resulting conceptual structure is:

```text
data/preprocessed/20250515/
├── <PATIENT_ID_1>/
│   ├── <image>.jpg
│   └── ...
├── <PATIENT_ID_2>/
│   ├── <image>.jpg
│   └── ...
└── ...
```

The notebook stores these values in an in-memory table named:

```text
patients_image_count
```

with the columns:

```text
pt_id
folder_path
n_images
```

In the bundled notebook execution:

```text
35 questionnaire participants were loaded
the median number of images per participant was 4
```

Some participants had substantially more images than the median, with a maximum of 16.

## Step 3: create the image manifest

The notebook parses all image filenames and creates an image-level table named:

```text
images_details
```

The table contains:

```text
patient_id
image_name
image_path
eye
image_type
```

### Column definitions

| Column       | Description                                                        |
| ------------ | ------------------------------------------------------------------ |
| `patient_id` | Pseudonymous participant identifier                                |
| `image_name` | Combination of participant identifier and image timestamp          |
| `image_path` | Environment-specific path or output stem associated with the image |
| `eye`        | `L` for left eye or `R` for right eye                              |
| `image_type` | `Color` or `OpacitySuppression`                                    |

The notebook saves the manifest as:

```text
20250515_imageList.csv
```

The output path is currently hard-coded:

```python
images_details.to_csv(
    "/storage/DSH/projects/inspired/data/20250515_imageList.csv",
    index=False,
)
```

Update this destination before running the notebook elsewhere.

## Step 4: create the AutoMorph resolution file

The final section of `explore_images.ipynb` creates a table containing:

```text
fundus
res
```

Each retinal-image filename is assigned a resolution value of:

```text
0.0039
```

The table is saved as:

```text
resolution_information.csv
```

The bundled version is located at:

```text
code/data_exploration/resolution_information.csv
```

A simplified implementation is:

```python
resolution_information = pd.DataFrame(
    {
        "fundus": images,
        "res": 0.0039,
    }
)

resolution_information.to_csv(
    "resolution_information.csv",
    index=False,
)
```

Verify that `0.0039` is appropriate for all images before applying the same value to new acquisitions.

## Step 5: run AutoMorph

Run the retinal images through AutoMorph using the generated resolution file.

The summary notebook expects two AutoMorph outputs:

```text
Results/M1/results_ensemble.csv
Results/M3/Macular_Features.csv
```

Their roles are:

```text
M1  image-quality assessment
M3  macular, optic-disc, and retinal vascular features
```

The AutoMorph output directories are not included in this repository.

## Step 6: combine AutoMorph results

Open:

```text
preprocessing/autoMorph_summary.ipynb
```

Configure the paths to:

```python
M1 = pd.read_csv(
    "/path/to/AutoMorph/Results/M1/results_ensemble.csv"
)

M3 = pd.read_csv(
    "/path/to/AutoMorph/Results/M3/Macular_Features.csv"
)

image_list = pd.read_csv(
    "/path/to/20250515_imageList.csv"
)
```

The notebook matches AutoMorph results to the INSPIRED image manifest using `image_name`.

For each image:

1. search for the corresponding M1 quality prediction;
2. skip the image when no M1 prediction is found;
3. retrieve the AutoMorph quality rating;
4. retrieve M3 features when the image is usable and an M3 result exists;
5. otherwise retain the image with missing feature values;
6. append image metadata, quality rating, and features to the combined table.

The resulting DataFrame is named:

```text
autoMorph_features
```

## Quality ratings

The notebook uses the following AutoMorph quality codes:

| Value | Interpretation |
| ----: | -------------- |
|   `0` | Good           |
|   `1` | Usable         |
|   `2` | Bad            |

Images with quality rating `2` are retained in the summary table, but all M3 retinal features are set to missing.

The bundled feature table contains:

```text
82 good images
34 usable images
46 bad images
```

Feature values are completely missing for:

```text
46 bad-quality images
4 good-quality images without matched M3 features
11 usable images without matched M3 features
```

A good or usable quality rating therefore does not guarantee that the M3 feature-extraction stage produced a result.

## Retinal features

The combined table includes optic-disc, optic-cup, and retinal vascular measurements.

### Optic-disc and optic-cup features

```text
Disc_height
Disc_width
Cup_height
Cup_width
CDR_vertical
CDR_horizontal
```

### Global vascular features

```text
Fractal_dimension
Vessel_density
Average_width
Distance_tortuosity
Squared_curvature_tortuosity
Tortuosity_density
```

Separate artery and vein versions of these measurements are also included.

### Zone-specific vascular features

Features are provided for AutoMorph zones B and C, including:

* fractal dimension;
* vessel density;
* average vessel width;
* distance tortuosity;
* squared-curvature tortuosity;
* tortuosity density;
* artery-specific measurements;
* vein-specific measurements;
* central retinal artery equivalent;
* central retinal vein equivalent;
* arteriovenous ratio.

The table contains Hubbard and Knudtson variants of:

```text
CRAE
CRVE
AVR
```

## Save the combined feature table

The current version of `autoMorph_summary.ipynb` builds the table in memory but does not contain an explicit save command.

Add the following cell after creating `autoMorph_features`:

```python
output_path = (
    "/storage/DSH/projects/inspired/data/preprocessed/"
    "20250515_autoMorph_features.csv"
)

autoMorph_features.to_csv(output_path, index=False)
```

For a portable relative destination:

```python
autoMorph_features.to_csv(
    "preprocessing/20250515_autoMorph_features.csv",
    index=False,
)
```

## Bundled data snapshot

The repository includes two derived CSV files.

### Resolution table

```text
File: code/data_exploration/resolution_information.csv
Rows: 192
Columns: 2
Unique image filenames: 192
Resolution values: 0.0039 for every image
```

### AutoMorph feature table

```text
File: preprocessing/20250515_autoMorph_features.csv
Rows: 162
Columns: 78
Participants: 34
Eyes: 90 right, 72 left
Image types: 160 Color, 2 OpacitySuppression
```

The resolution table contains 192 images, whereas the AutoMorph summary contains 162. The summary notebook skips images without a matched M1 quality-assessment result.

## Recommended analysis checks

Before using the resulting table for statistical analysis or machine learning, inspect:

* the number of images per participant;
* repeated images from the same eye;
* image-quality distributions;
* missing AutoMorph features;
* differences between left and right eyes;
* differences between colour and opacity-suppression images;
* feature distributions and outliers;
* consistency of image resolution;
* potential participant-level data leakage.

Because several images may belong to the same participant, train-validation-test splits must be performed at participant level rather than image level.

Images from the same participant must never be distributed across different evaluation subsets.

## Current limitations

The repository currently has several environment-specific or incomplete components:

1. Both notebooks contain absolute FBK storage paths.
2. The raw questionnaire and retinal images are not included.
3. AutoMorph and its outputs are not included.
4. The image-copying code uses the Unix `cp` command and is not directly portable to Windows.
5. The notebooks assume a specific underscore-separated filename convention.
6. The image manifest uses a path or stem that may not point directly to the copied JPEG file.
7. All images are assigned the same resolution value without checking image-specific metadata.
8. AutoMorph results are matched using substring searches rather than an explicit unique key.
9. The M1 matching code assumes that each image has exactly one matching prediction.
10. The summary notebook does not currently save its final DataFrame.
11. The repository contains no modelling or inferential-analysis code.
12. No dependency file is included.
13. No automated tests are included.
14. The `.gitignore` file is currently empty.
15. No licence file is included.

## Suggested improvements

Recommended next steps for making the workflow more portable and reproducible include:

* replace absolute paths with a configuration file or command-line arguments;
* replace `os.system("cp ...")` with `shutil.copy2`;
* validate filenames before parsing them;
* use exact image identifiers when merging tables;
* explicitly report unmatched M1 and M3 images;
* derive image resolution from acquisition metadata where possible;
* add participant-level quality-control summaries;
* save notebook outputs automatically;
* add a pinned Python environment;
* add unit tests for filename parsing and result merging;
* add a suitable `.gitignore`;
* add a licence before public distribution.

## Privacy and data governance

The repository processes biomedical images and questionnaire information.

The bundled feature table contains:

* pseudonymous participant identifiers;
* image identifiers;
* absolute internal storage paths;
* derived retinal measurements.

Do not publish or redistribute these data without verifying that this is permitted by:

* the study protocol;
* participant consent;
* institutional policies;
* ethics approvals;
* applicable privacy legislation.

Before making the repository public, consider removing or replacing internal paths and confirming whether the derived CSV files may be shared.

Do not commit:

* raw retinal images;
* identifiable questionnaire data;
* access credentials;
* private storage locations;
* participant re-identification keys.

Add an explicit licence before publicly distributing the repository. Without a licence, reuse, modification, and redistribution are not automatically permitted.
