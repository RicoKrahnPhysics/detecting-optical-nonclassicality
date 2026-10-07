# Moments classifier

Based on M. Jung's `IdentifyingOpticalNonclassicality` repository. The original
classifier files keep their layout. `moment_features.py` calculates raw photon
number moments; `raw_dataset.py` is shared by the two dataset scripts.

## Build datasets from TXT

The input directory for a run contains `weak_pnr_100000_*.txt` files. A raw
line has the file's unique ID, eight herald counts, eight output counts, one
EOM value and the occurrence count. The scripts use NumPy and h5py. From the
classifier directory, for example:

```bash
python build_interpol.py --run 1 --raw-dir /home/re36wir/data/Run1/Run1 --gap 6
python build_extrapol.py --run 1 --raw-dir /home/re36wir/data/Run1/Run1 --train-percent 80
```

Use `--out-dir PATH` to choose another `datasets` directory. Each pair is
written under `datasets/NAME/`; the scripts stop if NAME already exists.
Omit `--gap` or `--train-percent` to build all nine values for that run.
Run the script for each of Runs 1–8 for the full sweep.

Files with a malformed row, mixed EOM values or a total other than 100,000
shots are skipped. Contiguous acquisition periods allow up to four missing
file indices; 20 files (2 s) at each end are left out. A one-second bin uses
ten accepted files, followed by five skipped files (0.5 s). As in the earlier
preparation scripts, window positions are counted in the cleaned file
sequence. The three families use the same number of available bins.

Interpolation assigns one bin per family to train, followed by `gap` bins
to test, then repeats. Extrapolation takes the earliest requested fraction
of bins of each family for train and all later bins for test. In both cases
SBS keeps every non-vacuum herald with at least 100 counts. No GBS
analytical-eigenvalue filter is applied. The output
vacuum remains in the measured histograms. Original train states are kept;
the smaller of GBS/SBS is copied up to the larger family, then the smaller
label class is copied up to the other. Test states are never oversampled.

These builders write the data used by the classifier, not a new validation
file. Period IDs identify physical acquisition periods within a run. The
classifier splits complete periods from the train file for validation.
Train/test bins can still share a period, especially at an extrapolation
boundary or among the interleaved interpolation bins.
Some early extrapolation splits, particularly 10%, have too few training
periods for the classifier's grouped validation and cannot be trained.

## Run the classifier

Set `modes=8` in `hyperparams.py` for these experimental data and choose
one or two encoding layers. Then run:

```bash
python main_AlCla.py NAME 0.0 0.0
```

Only `NAME_train.h5` is used for fitting and validation; `NAME_test.h5` is
read after checkpoint selection. The first checkpoint with minimum loss on
distinct validation states is chosen. Copies are retained for fitting but
count only once in train metrics. At `h=0`, the zero-margin loss is zero for
both labels, so also check accuracy and the sign of `h`.

The features are ordinary raw moments, including diagonal `<n_i^2>` terms.
They are neither normally ordered nor `g`-normalised. Output vacuum events
enter the histogram normalisation even though their photon counts are zero.
