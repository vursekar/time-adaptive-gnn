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
                  'add_identity': True,
                  'time_modulation': False,
                  'time_mod_layers': 0,
                  'time_mod_activations':['relu','sigmoid'] ,
                  'l2_reg': 0.00}


class STGCN_Block(tf.keras.layers.Layer):

    def __init__(self, num_nodes, num_steps,
                       channels, kernel_size):

        super().__init__()

        self.num_nodes = num_nodes
        self.num_steps = num_steps
        self.channels = channels

        self.tgc1 = tf.keras.layers.Conv1D(filters=2*channels[0],kernel_size=(kernel_size,),
                                           activation='linear')

        self.tgc2 = tf.keras.layers.Conv1D(filters=2*channels[2],kernel_size=(kernel_size,),
                                           activation='linear')

        self.gc = tf.keras.layers.Conv2D(filters=channels[0],kernel_size=(1, 1),
                                         activation='relu')

    def call(self,x,A,batched_A):

        x = self.tgc1(x)
        x = tf.keras.activations.sigmoid(x[:,:,:,self.channels[0]:])*x[:,:,:,:self.channels[0]]

        if batched_A:
            x = tf.einsum('bnm,bmtd->bntd',A,x)
        else:
            x = tf.einsum('nm,bmtd->bntd',A,x)

        x = self.gc(x)
        x = self.tgc2(x)
        x = tf.keras.activations.sigmoid(x[:,:,:,self.channels[2]:])*x[:,:,:,:self.channels[2]]

        return x

class TimeModMLP(tf.keras.Model):

    def __init__(self, emb_dim, num_hidden, hidden_activation='relu', final_activation='sigmoid'):
        super().__init__()

        self.all_layers = []

        for i in range(num_hidden):
            self.all_layers.append(tf.keras.layers.Dense(emb_dim, activation=hidden_activation))

        self.all_layers.append(tf.keras.layers.Dense(emb_dim, activation=final_activation))

    def call(self, x):

        out = self.all_layers[0](x)

        for i in range(1, len(self.all_layers)):
            out = self.all_layers[i](out)

        return out


class STGCN(tf.keras.Model):

    def __init__(self, params=default_params):
        super().__init__()
        """
        Expected input: (batch, num_nodes, num_steps, input_features)
        """

        self.num_nodes = params['num_nodes']
        self.num_steps = params['num_steps']
        self.input_features = params['input_features']
        self.output_features = params['output_features']
        self.num_blocks = params['num_blocks']
        self.num_time_dims = params['num_time_dims']

        self.block_units = params['block_units']
        self.block_kernels = params['block_kernels']

        self.A = params['A']
        self.add_identity = params['add_identity']

        if self.add_identity:
            self.I = tf.eye(self.num_nodes)
        else:
            self.I = 0.

        self.adapt_adj = self.A is None
        self.emb_dim = params['emb_dim']
        self.time_modulation = params['time_modulation'] and (self.num_time_dims>0)
        self.time_mod_layers = params['time_mod_layers']
        self.time_mod_activations = params['time_mod_activations']

        if self.adapt_adj:

            self.E1 = tf.Variable(initial_value=tf.keras.initializers.RandomNormal(stddev=1)(shape=(self.num_nodes, self.emb_dim),
                                                                                   dtype="float32"),trainable=True)

            self.E2 = tf.Variable(initial_value=tf.keras.initializers.RandomNormal(stddev=1)(shape=(self.num_nodes, self.emb_dim),
                                                                                   dtype="float32"),trainable=True)

        if self.time_modulation:

            self.time_mod_reshaper = tf.keras.layers.Reshape((self.num_nodes, -1))
            self.time_mod1 = TimeModMLP(self.emb_dim, self.time_mod_layers, hidden_activation=self.time_mod_activations[0],
                                        final_activation=self.time_mod_activations[1])
            self.time_mod2 = TimeModMLP(self.emb_dim, self.time_mod_layers, hidden_activation=self.time_mod_activations[0],
                                        final_activation=self.time_mod_activations[1])


        self.block_layers = []
        final_kernel_size = self.num_steps

        for i in range(self.num_blocks):
            self.block_layers.append(STGCN_Block(self.num_nodes, self.num_steps, channels=self.block_units[i], kernel_size=self.block_kernels[i]))
            final_kernel_size -= 2*(self.block_kernels[i] - 1)

        self.tgc = tf.keras.layers.Conv1D(filters=self.block_units[i][2],kernel_size=(final_kernel_size,),
                                           activation='linear')

        self.fc = tf.keras.layers.Dense(self.output_features*self.num_steps, activation='linear')
        self.reshaper = tf.keras.layers.Reshape((self.num_nodes, self.num_steps, self.output_features))


    def call(self,x):
        """
        Our input shape is (batch, nodes, time, features)
        Expected that last few input dimensions are time features
        """

        if self.time_modulation and self.adapt_adj:
            time_input = x[:,:,-1,-self.num_time_dims:]
            x = x[:,:,:,:-self.num_time_dims]

            time_input1 = tf.einsum('bnk, nd->bnkd',time_input,self.E1)
            time_input2 = tf.einsum('bnk, nd->bnkd',time_input,self.E2)

            time_input1 = self.time_mod_reshaper(time_input1)
            time_input2 = self.time_mod_reshaper(time_input2)

            E1 = self.E1 * self.time_mod1(time_input1)
            E2 = self.E2 * self.time_mod2(time_input2)

            self.A = self.I + softmax(relu(tf.matmul(E1, E2, transpose_b=True)),axis=1)

        elif self.adapt_adj:

            self.A = self.I + softmax(relu(tf.matmul(self.E1, self.E2, transpose_b=True)),axis=1)

        else:
            pass

        for i in range(self.num_blocks):
            x = self.block_layers[i](x, self.A, batched_A=self.time_modulation)

        x = self.tgc(x)
        x = self.fc(x)
        x = self.reshaper(x)

        return x
