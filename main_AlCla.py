import numpy as np
import jax, h5py, sys, os
import jax.numpy as jnp
from absl import flags, app
from flax import linen as nn
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import hyperparams
FLAGS = flags.FLAGS
app.parse_flags_with_usage(['.'])

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

from train_AlCla import *

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = "serif"

# States from the same measurement period may share source drift even if their
# heralding patterns differ. Keep complete periods in validation so the
# classifier is checked on measurements it did not fit.

def load_train_periods(current_ds:str)->(dict,dict):
    DATASET_PATH = "datasets/" + current_ds
    file = DATASET_PATH + f'/{current_ds}_train.h5'
    data = read_states(file)
    train_ids, uniq, val_rows = split_periods(data, FLAGS.validation_fraction)
    size = FLAGS.modes + FLAGS.modes**2
    if FLAGS.num_encode_layers == 2:
        size += FLAGS.modes**3
    if data['images'].shape[1] != size:
        raise ValueError(f'{file}: expected {size} moment entries per state, got {data["images"].shape[1]}')
    def select(data, ids):
        return {'images': data['images'][ids], 'labels': data['labels'][ids]}
    splits = {'train': select(data, train_ids),
              'train_unique': select(data, uniq),
              'validation': select(data,val_rows)}
    ids = {'train': train_ids, 'train_unique': uniq,
           'validation': val_rows}
    return splits, ids, set(data['keys'])

def read_test_states(current_ds:str, train_keys:set) -> (dict,np.array):
    file = 'datasets/' + current_ds + f'/{current_ds}_test.h5'
    data = read_states(file)
    overlap = train_keys & set(data['keys'])
    if overlap:
        raise ValueError(f'{file}: {len(overlap)} states also occur in train; first: {next(iter(overlap))}')
    size = FLAGS.modes + FLAGS.modes**2
    if FLAGS.num_encode_layers == 2:
        size += FLAGS.modes**3
    if data['images'].shape[1] != size:
        raise ValueError(f'{file}: expected {size} moment entries per state, got {data["images"].shape[1]}')
    ids = data['unique']
    return {'images': data['images'][ids], 'labels': data['labels'][ids]}, ids

def init_classifier():
    if FLAGS.num_encode_layers not in (1, 2):
        raise ValueError(f'Raw moments support 1 or 2 encoding layers, got {FLAGS.num_encode_layers}')
    init_x = symbols(f'x:{FLAGS.modes*(FLAGS.num_encode_layers+1)}', positive=True)
    symb_expression, permu, (decoder_fn, try_theta_init) = jaxdecoder(init_x,FLAGS.modes,FLAGS.num_encode_layers)
    print(f'Theta init is {try_theta_init} and the reshuffled theta is {np.array(try_theta_init)[np.array(permu)]}\n')
    print('The used permu is', np.array(permu))
    QuAttnNet = QuantumAttentionNet(num_modes=FLAGS.modes, 
                                        num_layers=FLAGS.num_encode_layers,
                                        decoder_fn=decoder_fn)
    
    opt, state, params = create_train_state(init_rng, QuAttnNet,
                            num_modes=FLAGS.modes,
                            num_layers=FLAGS.num_encode_layers,
                            learning_rate=FLAGS.learning_rate)
    return opt, state, params, permu, QuAttnNet


def save_period_split(current_ds:str, ids:dict):
    SAVENAME = 'saved_params/' + current_ds + '/split_' + current_ds + '.hdf5'
    os.makedirs(os.path.dirname(SAVENAME), exist_ok=True)
    with h5py.File(SAVENAME, "w") as f:
        for name, values in ids.items():
            f.create_dataset(name, data=values)
        f.attrs['validation_fraction'] = FLAGS.validation_fraction
        f.attrs['seed'] = 0
    print('Saved split indices in:', SAVENAME)
    return 0

# Fit with the supplied oversampling, but compare checkpoints on unique
# physical states. That way a copied histogram does not decide which
# polynomial appears to generalize best to a held-out measurement period.

def fit_periods(rng,opt,state,params,QuAttnNet, epochs, train_ds, train_unique, val_ds,regularization, suppress_k):
    train_loss, train_acc = train_test_epoch(opt, state, QuAttnNet,params,train_unique, rng,
                                           'test', regularization, suppress_k)
    val_loss, val_acc = train_test_epoch(opt, state, QuAttnNet,params,val_ds, rng,
                                       'test', regularization, suppress_k)
    best_loss = float(val_loss)
    best_epoch = 0
    best_params = jax.tree_util.tree_map(lambda x: np.array(x), params)
    points = [0]
    train_losses = [float(train_loss)]
    val_losses = [float(val_loss)]

    for epoch in range(1, epochs + 1):
        rng, input_rng = jax.random.split(rng)

        state, params, _, _ = train_test_epoch(opt, state, QuAttnNet,params,train_ds, input_rng,
                                            'train', regularization, suppress_k)
        if epoch % 5 == 0 or epoch == epochs:
            train_loss, train_acc = train_test_epoch(opt, state, QuAttnNet,params,train_unique, input_rng,
                                                   'test', regularization, suppress_k)
            val_loss, val_acc = train_test_epoch(opt, state, QuAttnNet,params,val_ds, input_rng,
                                               'test', regularization, suppress_k)
            points.append(epoch)
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            loss = float(val_loss)
            # Keep the first minimum of validation loss, including epoch 0.
            # Ties do not replace it; the test set plays no part in this choice.
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_params = jax.tree_util.tree_map(lambda x: np.array(x), params)
            print(f'epoch {epoch} train accuracy: {train_acc}, validation accuracy: {val_acc}, validation loss: {val_loss}')
    print(f'Finished with epochs!')
    return best_params, best_epoch, params, points, train_losses, val_losses

def save_checkpoint(file:str, params:dict, epoch:int):
    with h5py.File(file, 'w') as f:
        f.attrs['epoch'] = epoch
        for name, value in params.items():
            f.create_dataset(name, data=np.asarray(value))

# The supplied train file provides fitting states and held-out periods.
# Only after validation fixes the checkpoint do we read the separate test
# file, so its labels cannot affect the chosen optical classifier.

if __name__ == "__main__":
    Current_dataset= sys.argv[1]
    regularization = float(sys.argv[2])
    if len(sys.argv) < 4:
        suppress_k = 0.0
    else:
        suppress_k = float(sys.argv[3])
    print('the regularization is: {}'.format(regularization))
    print('the suppressing factor is: {}'.format(suppress_k))

    splits, ids, train_keys = load_train_periods(Current_dataset)
    save_period_split(Current_dataset, ids)
    opt, state, params, permu, QuAttnNet = init_classifier()
    
    print(f'parameters have shapes:', jax.tree_util.tree_map(lambda x: x.shape, params))

    best_params, best_epoch, params, points, loss_train, loss_val = fit_periods(
                                                        rng, 
                                                        opt, 
                                                        state, 
                                                        params, 
                                                        QuAttnNet,
                                                        FLAGS.epochs, 
                                                        splits['train'],
                                                        splits['train_unique'],
                                                        splits['validation'],
                                                        regularization,
                                                        suppress_k)
    OUTPUT_PATH = 'saved_params/' + Current_dataset + '/'
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    save_checkpoint(OUTPUT_PATH + 'best_validation.h5', best_params, best_epoch)
    save_checkpoint(OUTPUT_PATH + 'final.h5', params, FLAGS.epochs)
    test_ds, ids['test'] = read_test_states(Current_dataset, train_keys)
    save_period_split(Current_dataset, ids)
    best_loss, best_acc = train_test_epoch(opt, state, QuAttnNet,best_params,test_ds, rng,
                                          'test', regularization, suppress_k)
    final_loss, final_acc = train_test_epoch(opt, state, QuAttnNet,params,test_ds, rng,
                                            'test', regularization, suppress_k)
    print(f'best validation epoch: {best_epoch}')
    print(f'best checkpoint test accuracy: {best_acc}, test loss: {best_loss}')
    print(f'final checkpoint test accuracy: {final_acc}, test loss: {final_loss}')

    plt.plot(points,loss_train, label='training dataset')
    plt.plot(points,loss_val, label='validation dataset')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    savename=f'Loss_curve_{Current_dataset}_{FLAGS.num_encode_layers}el_{FLAGS.shots}kshots'
    #plt.savefig(savename +'.pdf')
    #plt.savefig(savename +'.svg')
    plt.show()
