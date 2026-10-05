import numpy as np


def raw_moments(images, occ, num_encode_layers):
    images = np.asarray(images, dtype=np.float64)
    occ = np.asarray(occ, dtype=np.float64)

    if images.ndim != 3 or occ.shape != (images.shape[0], images.shape[2]):
        raise ValueError(f"images shape {images.shape}, occ shape {occ.shape}; expected matching states and patterns")
    if np.any(occ < 0) or not np.all(np.isfinite(occ)) or np.any(occ.sum(axis=-1) == 0):
        raise ValueError("occ contains negative/nonfinite counts or an empty state")

    # A row lists photon-number patterns, with occ giving how often each pattern
    # was measured. Empty slots are just padding and may even contain NaNs.
    # They must contribute neither a photon number nor a histogram weight.
    images = np.where(occ[:, None, :] > 0, images, 0)
    if np.any(images < 0) or not np.all(np.isfinite(images)):
        raise ValueError("active photon patterns contain negative or nonfinite counts")
    w=occ / occ.sum(axis=-1, keepdims=True)

    # Normalize by all measured events, including vacuum events. Vacuum adds
    # zero to the numerators but still matters for the expectation value.
    # These are raw moments <n_i>, <n_i n_j> and, if needed, <n_i n_j n_k>.
    # On the diagonal this means <n_i^2>, not a factorial moment or g^(2).
    m1 = np.einsum("bip,bp->bi", images, w)
    m2 = np.einsum("bip,bjp,bp->bij", images, images, w, optimize=True)
    parts = [m1.reshape(len(images), -1), m2.reshape(len(images), -1)]

    if num_encode_layers > 1:
        m3 = np.einsum("bip,bjp,bkp,bp->bijk",
                       images, images, images, w, optimize=True)
        parts.append(m3.reshape(len(images), -1))
    return np.concatenate(parts, axis=-1)
