from absl import flags, app

# encoder
flags.DEFINE_integer('num_encode_layers', default=1,  help='number of layers')
#flags.DEFINE_integer('hidden_dim', default=1,  help='hidden dimension')

# dense decoder
#flags.DEFINE_boolean('dense_decoder', default=False, help='Implement dense decoder')
flags.DEFINE_list('layer_dims', default=['14','4','1'],  help='hidden dimension in the dense decoder')

# algebraic decoder
flags.DEFINE_boolean('diagonal',default=False,  help='Implement a diagonal encoder')
flags.DEFINE_boolean('check_indv_modes_first', default=False,  help='Implement a decision rule that examines individual mode contributions first')

# data
flags.DEFINE_integer('modes',default=1,  help='number of modes')
flags.DEFINE_integer('shots', default=1000,help='number of measurement shots = M')

# training
flags.DEFINE_integer('epochs', default=900, help='number of epochs')
flags.DEFINE_integer('batch_size', default=39, help='size of the minibatches')
flags.DEFINE_float('learning_rate', default=1e-2,help='learning rate of the adam optimizer')
flags.DEFINE_float('validation_fraction', default=0.2, help='fraction of periods used for validation')

FLAGS = flags.FLAGS
app.parse_flags_with_usage(['.'])
