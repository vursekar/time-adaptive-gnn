import tensorflow as tf
from tensorflow.keras.activations import relu, softmax


default_params = {'num_nodes': 300,
                  'num_steps': 10,
                  'input_features': 3,
                  'output_features': 2,
                  'num_time_dims': 1,
                  'num_blocks': 2,
                  'block_units': [[64,16,64], [32,16,32]],
                  'block_kernels': [3,3],
                  'A': None,
                  'emb_dim': 32,
                  'time_modulation': False,
                  'time_mod_layers': 0,
                  'time_mod_activations':['relu','sigmoid'] ,
                  'l2_reg': 0.00}


class AGCRN(tf.keras.Model):

    def __init__(self, params):

        self.num_nodes = params['num_nodes']
        self.num_steps = params['num_steps']
        self.input_features = params['input_features']
        self.output_features = params['output_features']
        self.num_time_dims = params['num_time_dims']

        self.emb_dim = params['emb_dim']

        self.E1 = tf.Variable(initial_value=tf.keras.initializers.RandomNormal(stddev=1)(shape=(self.num_nodes, self.emb_dim),
                                                                                   dtype="float32"),trainable=True)
        
