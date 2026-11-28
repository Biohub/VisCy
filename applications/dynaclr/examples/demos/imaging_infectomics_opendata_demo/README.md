# Infectomics Imaging open-data demo

`get-to-know-infectomics-imaging.ipynb` is the tutorial for
[Infectomics Imaging](https://registry.opendata.aws/infectomics-imaging/), the live-cell
imaging atlas in the public bucket `s3://bh-infectomics` (us-east-1).

The atlas holds 62 OME-Zarr acquisitions of live A549 cells. Each well carries one
GFP-tagged organelle or cell-state marker, imaged through Dengue, Zika and herpes simplex
virus infection and through four drug treatments, on the Mantis light-sheet microscope
(every 30 minutes) and an Andor Dragonfly spinning-disk confocal (every 6 hours).

The notebook streams everything from S3 with anonymous HTTP range requests. It downloads
no store in full, needs no AWS credentials, and runs in about two minutes. It has four parts:

1. What is in the bucket: the stores and their per-field-of-view curation tables.
2. The single-cell atlas: the DynaCLR embedding behind the interactive explorer.
3. What the images look like: organelles across infections, and one infection in time.
4. Open challenge: predicting the course of an infection from morphology.

## Requirements

```bash
pip install "boto3>=1.34" "fsspec[http]>=2024" "zarr>=3" "pyarrow>=14" "numpy>=1.26" "matplotlib>=3.8"
```

The outputs are committed as executed.
