import jax, optax, csv, functools, h5py
import jax.numpy as jnp
from absl import flags, app
import numpy as np
from jax import random
from functools import partial
from flax import linen as nn
import optax.tree
from optax import contrib
from jax.scipy.special import xlog1py, xlogy

import hyperparams
FLAGS = flags.FLAGS
app.parse_flags_with_usage(['.'])

from model_AlCla import *
from moment_features import raw_moments

# HYPERPARAMETERS
PATIENCE = 10 # Number of epochs with no improvement after which learning rate will be reduced
COOLDOWN = 0 # Number of epochs to wait before resuming normal operation after the learning rate reduction
FACTOR = 0.5 # Factor by which to reduce the learning rate
RTOL = 1e-4 # Relative tolerance for measuring the new optimum
ACCUMULATION_SIZE = 39 # Number of iterations to accumulate an average value


def read_states(file:str) -> dict:
    with h5py.File(file, 'r') as data:
        for name in ('images', 'occ', 'labels', 'meta/state_id', 'meta/kind',
                     'meta/heralding_pattern', 'meta/period_id', 'meta/global_bin',
                     'meta/start_file_index', 'meta/end_file_index'):
            if name not in data:
                raise ValueError(f'{file}: missing {name}')
        labels = data['labels'][:]
        if len(labels) == 0 or not np.all(np.isin(labels, [0, 1])):
            raise ValueError(f'{file}: labels must be 0 or 1 and the file must contain states')
        meta = data['meta']
        names = meta['state_id'].asstr()[:]
        kinds = meta['kind'].asstr()[:]
        patterns = meta['heralding_pattern'].asstr()[:]
        periods = meta['period_id'][:]
        bins = meta['global_bin'][:]
        starts = meta['start_file_index'][:]
        ends = meta['end_file_index'][:]
        # If no source ID exists, equal period numbers stay grouped conservatively.
        source = meta['source_id'].asstr()[:] if 'source_id' in meta else ['']*len(labels)
        chunks = []
        for start in range(0, len(labels), 128):
            end = start + 128
            try:
                chunks.append(raw_moments(data['images'][start:end], data['occ'][start:end],
                                          FLAGS.num_encode_layers))
            except ValueError as exc:
                raise ValueError(f'{file}, rows {start}:{min(end,len(labels))}: {exc}') from exc
    images = np.concatenate(chunks)
    keys, groups, unique, seen = [], [], [], {}
    for idx in range(len(labels)):
        # Oversampled copies give a state more weight during fitting, but do
        # not represent new measurements. Strip their suffix so validation
        # and reported scores count that physical state only once.
        name = names[idx].split('__oversample', 1)[0]
        key = (source[idx], kinds[idx], name, patterns[idx], int(periods[idx]),
               int(bins[idx]), int(starts[idx]), int(ends[idx]))
        keys.append(key)
        groups.append((source[idx], int(periods[idx])))
        if key in seen:
            j = seen[key]
            if labels[idx] != labels[j] or not np.array_equal(images[idx], images[j]):
                raise ValueError(f'{file}: state {name} differs between rows {j} and {idx}')
        else:
            seen[key] = idx
            unique.append(idx)
    return {'images': images, 'labels': labels, 'keys': keys,
            'groups': groups, 'unique': np.asarray(unique)}

def split_periods(ds, fraction, seed=0):
    if not 0 < fraction < 0.5:
        raise ValueError(f'Validation fraction {fraction} must be between 0 and 0.5')
    unique = ds['unique']
    groups = []
    for idx in unique:
        group = ds['groups'][idx]
        if group not in groups:
            groups.append(group)
    if len(groups) < 3:
        raise ValueError(f'Only {len(groups)} measurement periods; need at least three')
    # Herald variants and nearby states from one acquisition period can share
    # slow source drift. Hold the whole period out so this correlation cannot
    # make validation look easier than a genuinely separate measurement.
    # Try seeded choices and keep one close to the requested state fraction
    # and class mix, with both labels present on each side.
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(200):
        count = max(1, round(len(groups)*fraction))
        held_out = {groups[i] for i in rng.permutation(len(groups))[:count]}
        fit_unique, val = [], []
        for idx in unique:
            if ds['groups'][idx] in held_out:
                val.append(idx)
            else:
                fit_unique.append(idx)
        if len(set(ds['labels'][val])) != 2 or len(set(ds['labels'][fit_unique])) != 2:
            continue
        gap=abs(len(val)/len(unique)-fraction) + abs(ds['labels'][val].mean()-ds['labels'][unique].mean())
        if best is None or gap < best[0]:
            best = (gap, fit_unique, val, held_out)
    if best is None:
        counts = np.bincount(ds['labels'][unique].astype(int), minlength=2)
        raise ValueError(f'No period split with both classes (unique labels: {counts.tolist()}, periods: {len(groups)})')
    train = []
    # Retain training copies for their intended weighting; val uses unique
    # state indices, so repeated histograms do not inflate its score.
    for idx in range(len(ds['labels'])):
        if ds['groups'][idx] not in best[3]:
            train.append(idx)
    return np.asarray(train), np.asarray(best[1]), np.asarray(best[2])

def create_train_state(key:jnp.array, QuAttnNet, num_modes:int, num_layers:int, learning_rate:float):
    params = QuAttnNet.init_params(key,num_modes,num_layers)
    opt = optax.chain(optax.adam(learning_rate),
            contrib.reduce_on_plateau(
                patience=PATIENCE,
                cooldown=COOLDOWN,
                factor=FACTOR,
                rtol=RTOL,
                accumulation_size=ACCUMULATION_SIZE,
            ),
        )
    opt_state = opt.init(params)
    return opt, opt_state, params

@partial(jax.jit, static_argnames=['QuAttnNet','regularization_strength', 'suppressing_k'])
def loss_fn(params:dict, 
        images:jnp.array, 
        label:jnp.array, 
        QuAttnNet,
        regularization_strength:float, 
        suppressing_k:float) -> (jnp.array, jnp.array):
        corrs = QuAttnNet.Encoder(params, images)
        h = QuAttnNet.classifier_value(params, corrs)
        logits = QuAttnNet.AlgebraicDecoder(params, corrs)
        # The optical state is labelled classical (0) or nonclassical (1).
        # At zero margin, a classical state pays only for h < 0; a
        # nonclassical one pays only for h > 0. The sign of h is the boundary
        # before the sigmoid. A state exactly at h = 0 has zero hinge loss
        # for either label, so inspect validation accuracy as well as loss.
        loss = jnp.mean((1.-label)*jnp.maximum(0., -h) + label*jnp.maximum(0., h))
        reg = jnp.sum((1.-label)*(logits**2))
        loss += regularization_strength*reg
        for i in range(FLAGS.num_encode_layers):
            loss += suppressing_k*jnp.sum((params[f'k{i}'])**2)
        return loss, logits

@partial(jax.jit, static_argnames=['QuAttnNet','regularization_strength', 'suppressing_k'])
def apply_model(params:dict, 
        images:jnp.array, 
        label:jnp.array, 
        QuAttnNet, 
        regularization_strength:float, 
        suppressing_k:float) -> (jnp.array, jnp.array,jnp.array,jnp.array):
    " compute gradients, zero-margin loss and accuracy "
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(params, images, label, QuAttnNet, regularization_strength, suppressing_k)
    accuracy = ((logits > 0.5) == label)
    return grads, loss, accuracy, logits

@partial(jax.jit, static_argnames=['opt'])
def update_model(params:dict, opt, opt_state, grads:jnp.array, loss:jnp.array):
    updates, opt_state = opt.update(grads, opt_state, params, value=loss)
    params = optax.apply_updates(params, updates)
    return params, opt_state

def train_test_epoch(opt, state, 
                    QuAttnNet, 
                    params:dict, 
                    train_ds:dict, 
                    rng:jnp.array, 
                    phase:str, 
                    regularization_strength:float, 
                    suppressing_k:float):
    """ If phase =='test'
            output: loss, accuracy
        If phase =='train'
            output: state, loss, accuracy
    """
    train_ds_size = len(train_ds['images'])
    perm = jax.random.permutation(rng, train_ds_size)
    images_reshuffled = jnp.array(train_ds['images'])[perm]
    labels_reshuffled = jnp.array(train_ds['labels'])[perm]
    
    epoch_loss, epoch_accuracy, test_loss, test_accuracy = [],[],[],[]
    
    v_apply = jax.vmap(apply_model, in_axes=(None,0,0,None,None,None), out_axes=0)
    if phase == 'train':
        for start in range(0, train_ds_size, FLAGS.batch_size):
            img = images_reshuffled[start:start+FLAGS.batch_size]
            labels_batch = labels_reshuffled[start:start+FLAGS.batch_size]
            grads, loss, accuracy, logits = v_apply(params, img, labels_batch, QuAttnNet,regularization_strength, suppressing_k)
            # average gradients
            average_over_batch = partial(jnp.mean, axis=0)
            grads_mean = jax.tree.map(average_over_batch, grads)          
            params, state = update_model(params, opt, state, grads_mean, jnp.mean(loss))

            # Clip k matrices with upper triangular
            if FLAGS.modes > 1:
                lower_k = -10.*jnp.triu(jnp.ones((FLAGS.modes, FLAGS.modes)))
                upper_k = 10.*jnp.triu(jnp.ones((FLAGS.modes, FLAGS.modes)))
            else:
                lower_k = -10.*jnp.ones((FLAGS.modes, FLAGS.modes)) # lower bound the K-values by -10
                upper_k = 10*jnp.ones((FLAGS.modes, FLAGS.modes))
            for l in range(FLAGS.num_encode_layers):
                params[f'k{l}'] = optax.projections.projection_box(params[f'k{l}'], lower=lower_k, upper=upper_k)

            # Clip amplitude and theta  
            low_amplify = 1.0
            up_amplify = 50.0
            params['amplify'] = optax.projections.projection_box(params['amplify'], lower=low_amplify, upper=up_amplify)
            lower_theta = jnp.array([-10.0]*len(params['theta']))
            upper_theta = jnp.array([ 10.0]*len(params['theta']))
            params['theta'] = optax.projections.projection_box(params['theta'], lower=lower_theta, upper=upper_theta)
            
            epoch_loss.append(jnp.sum(loss))
            epoch_accuracy.append(jnp.sum(accuracy))
        train_loss = np.sum(epoch_loss)/train_ds_size
        train_accuracy = np.sum(epoch_accuracy)/train_ds_size
        return state, params, train_loss, train_accuracy
    
    elif phase == 'test':
        for idx, img in enumerate(images_reshuffled):
            loss, logits = loss_fn(params, img, labels_reshuffled[idx], QuAttnNet,
                                   regularization_strength, suppressing_k)
            accuracy = ((logits > 0.5) == labels_reshuffled[idx])
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)
        test_loss = np.mean(epoch_loss)
        test_accuracy = np.mean(epoch_accuracy)
        return test_loss, test_accuracy    

def get_prediction_distribution(QuAttnNet, 
                            params:dict, 
                            train_ds:dict, 
                            regularization_strength:float, 
                            suppressing_k:float) -> (jnp.array,jnp.array):
    classicals_prediction = []
    non_classicals_prediction = []
    labels = jnp.array(train_ds['labels'])
    _, logits = loss_fn(params, jnp.array(train_ds['images']), labels, QuAttnNet,regularization_strength, suppressing_k)
    classical_indices = jnp.where((labels<1))
    nonclassical_indices = jnp.where((labels>0))
    comp_classicals_prediction = logits[classical_indices]
    comp_nonclassical_prediction = logits[nonclassical_indices]

    for i in range(len(train_ds['images'])):
        label = jnp.array(train_ds['labels'])[i]
        if label == 0:
            _, logits = loss_fn(params, train_ds['images'][i], 0, QuAttnNet,regularization_strength, suppressing_k)
            classicals_prediction.append(jnp.mean(logits))
        else:
            _, logits = loss_fn(params, train_ds['images'][i], 1, QuAttnNet,regularization_strength, suppressing_k)
            non_classicals_prediction.append(jnp.mean(logits))
    return jnp.array(classicals_prediction), jnp.array(non_classicals_prediction)

def get_predictions_and_encoder_outputs(QuAttnNet,
                                        best_params:dict, 
                                        train_ds:dict, 
                                        regularization_strength:float, 
                                        suppressing_k:float) -> (jnp.array,jnp.array,jnp.array):
    encoder_outputs = []
    predictions = []
    true_labels = []
    QuAttnNet = PreSymbolicRegressionNet(M=FLAGS.shots, num_modes=FLAGS.modes, num_layers=FLAGS.num_encode_layers)
    for i in range(len(train_ds['images'])):
        true_labels.append(jnp.array(train_ds['labels'])[i])
        y = QuAttnNet.Encoder(best_params, train_ds['images'][i])
        encoder_outputs.append(y[:,0])
        logits_unnormalized = QuAttnNet.DenseDecoder(best_params,y)
        predictions.append(jnp.mean(logits_unnormalized))
    return jnp.array(encoder_outputs), jnp.array(predictions), jnp.array(true_labels)
