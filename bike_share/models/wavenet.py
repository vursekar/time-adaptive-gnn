import tensorflow as tf


default_params = {'num_nodes': 300,
                  'num_steps': 10,
                  'input_features': 3,
                  'output_features': 2,
                  'num_time_dims': 1,
                  'emb_dim': 32,
                  'num_blocks': 4,
                  'num_layers': 2,
                  'res_channels': 32,
                  'skip_channels': 32,
                  'dilation_channels': 32,
                  'end_channels': 32,
                  'time_modulation': False,
                  'time_mod_layers': 0,
                  'time_mod_activations': ['relu','sigmoid'],
                  'kernel_size': 2,
                  'use_gc': True
                  }

class GraphConv(tf.keras.layers.Layer):

    def __init__(self, out_dim):
        super().__init__()

        self.out_dim = out_dim
        self.W = tf.keras.layers.Conv2D(out_dim, kernel_size=(1, 1))

    def forward(self, x, A, batched_A):

        if batched_A:
            out = tf.einsum('bnm,bmtd->bntd',A,x)
        else:
            out = tf.einsum('nm,bmtd->bntd',A,x)

        out = self.W(out)
        return out

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

class GraphWaveNet(tf.keras.Model):

    def __init__(self,params=default_params):
        """
        Our input shape is (batch, nodes, time, features)
        their inputs was (batch, features, nodes, time)
        """

        super().__init__()

        self.num_nodes = params['num_nodes']
        self.num_steps = params['num_steps']
        self.input_features = params['input_features']
        self.output_features = params['output_features']
        self.num_time_dims = params['num_time_dims']

        self.final_feature_dim = self.num_steps*self.output_features

        self.num_blocks = params['num_blocks']
        self.num_layers = params['num_layers']
        self.depth = self.num_blocks*self.num_layers
        self.use_gc = params['use_gc']

        self.res_channels = params['res_channels']
        self.skip_channels = params['skip_channels']
        self.dilation_channels = params['dilation_channels']
        self.end_channels = params['end_channels']
        self.kernel_size = params['kernel_size']

        self.emb_dim = params['emb_dim']
        self.time_modulation = params['time_modulation']
        self.time_mod_layers = params['time_mod_layers']
        self.time_mod_activations = params['time_mod_activations']

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


        self.lt = tf.keras.layers.Conv2D(filters=self.res_channels,kernel_size=(1,1))

        if not self.use_gc:
            self.res_convs = [tf.keras.layers.Conv1D(filters=self.res_channels, kernel_size=1) for _ in range(self.depth)]

        self.skip_convs = [tf.keras.layers.Conv1D(filters=self.skip_channels, kernel_size=1) for _ in range(self.depth)]
        self.graph_convs = [GraphConv(self.res_channels) for _ in range(self.depth)]
        self.bns = [tf.keras.layers.BatchNormalization() for _ in range(self.depth)]

        self.filter_convs = []
        self.gate_convs = []
        receptive_field = 1

        for b in range(self.num_blocks):

            scope = self.kernel_size - 1
            dilation = 1

            for l in range(self.num_layers):

                self.filter_convs.append(tf.keras.layers.Conv2D(self.dilation_channels,kernel_size=(1,self.kernel_size),
                                                               dilation_rate=dilation, activation='tanh'))
                self.gate_convs.append(tf.keras.layers.Conv2D(self.dilation_channels,kernel_size=(1,self.kernel_size),
                                                               dilation_rate=dilation, activation='sigmoid'))
                dilation = 2*dilation
                receptive_field += scope
                scope = 2*scope

        self.receptive_field = receptive_field
        self.lt_final1 = tf.keras.layers.Conv2D(self.end_channels, kernel_size=(1,1))
        self.lt_final2 = tf.keras.layers.Conv2D(self.final_feature_dim, kernel_size=(1,1))
        self.reshaper = tf.keras.layers.Reshape((self.num_nodes,self.num_steps,self.output_features))

    def call(self, x):

        num_steps = x.shape[2]

        if num_steps < self.receptive_field:
            paddings = tf.constant([[0,0],[0,0],[self.receptive_field-num_steps,0],[0,0]])
            x = tf.pad(x, paddings)

        x = self.lt(x)
        skip = 0


        if self.time_modulation:
            time_input = x[:,:,-1,-self.num_time_dims:]
            x = x[:,:,:,:-self.num_time_dims]

            time_input1 = tf.einsum('bnk, nd->bnkd',time_input,self.E1)
            time_input2 = tf.einsum('bnk, nd->bnkd',time_input,self.E2)

            time_input1 = self.time_mod_reshaper(time_input1)
            time_input2 = self.time_mod_reshaper(time_input2)

            E1 = self.E1 * self.time_mod1(time_input1)
            E2 = self.E2 * self.time_mod2(time_input2)

            self.A = tf.keras.activations.softmax(tf.keras.activations.relu(tf.matmul(E1, E2,transpose_b=True)),axis=1)

        else:
            self.A = tf.keras.activations.softmax(tf.keras.activations.relu(tf.matmul(self.E1, self.E2,transpose_b=True)),axis=1)


        for i in range(self.depth):

            residual = x
            fltr = self.filter_convs[i](residual)
            gate = self.gate_convs[i](residual)
            x = fltr*gate

            s = self.skip_convs[i](x)

            if i>0:
                skip = skip[:, :, -s.shape[2]:, :]
            else:
                skip = 0

            skip = skip + s

            if self.use_gc:
                graph_out = self.graph_convs[i](x, self.A, batched_A=self.time_modulation)
                x = x + graph_out
            else:
                x = self.res_convs[i](x)

            x = x + residual[:, :, -x.shape[2]:, :]

            x = self.bns[i](x)

        x = tf.keras.activations.relu(skip)
        x = tf.keras.activations.relu(self.lt_final1(x))
        x = self.lt_final2(x)
        x = tf.squeeze(x,axis=-2)
        x = self.reshaper(x)

        return x
