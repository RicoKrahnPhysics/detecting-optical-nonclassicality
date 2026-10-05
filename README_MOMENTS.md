# Moments Classifier

Based on M. Jung's `IdentifyingOpticalNonclassicality` repository. The structure and formatting of the original files have been preserved. Only `moment_features.py` was added as a small helper file.

## Data

Two files are expected in `datasets/NAME/`:

- `NAME_train.h5`
- `NAME_test.h5`

They contain `images` (states, modes, count patterns), `occ` (states, count patterns), `labels` (0 for classical, 1 for nonclassical), and the metadata fields `meta/state_id`, `meta/kind`, `meta/heralding_pattern`, `meta/period_id`, `meta/global_bin`, `meta/start_file_index`, and `meta/end_file_index`. This matches the output of the existing dataset builder. Each pattern frequency in `occ` contributes to the ordinary raw moments; the output vacuum is retained.

## Validation and test

Only `NAME_train.h5` is used for training and validation. Complete measurement periods are selected for validation with seed 0 (by default, 20% of the periods). All heralding variants and oversampled copies from a period stay together. Both subsets must contain classical and nonclassical states. Oversampled copies remain in the training data; for training and validation metrics, each physical state is counted once.

Epoch 0, every fifth epoch, and the final epoch are evaluated. The first checkpoint with the lowest validation loss is selected. Only then is `NAME_test.h5` loaded. The selected and final checkpoints are each evaluated once on the unique test states. `saved_params/NAME/` contains `best_validation.h5`, `final.h5`, and the indices used for the split in `split_NAME.hdf5`.

## Running the classifier

`hyperparams.py` contains the number of modes, the number of layers (1 or 2), shots, batch size, learning rate, number of epochs, and validation fraction. Run it as in Martina’s repository:

```bash
python main_AlCla.py NAME 0.0 0.0
```

The two numbers set the regularization strength and K suppression. The zero-margin loss is `max(0,-h)` for classical states and `max(0,h)` for nonclassical states. At `h=0`, the loss is zero for both classes, so accuracy and the sign of `h` should also be checked. The moments are neither normally ordered nor `g`-normalized.

