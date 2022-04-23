import tensorflow as tf

default_params = {'num_layers': 2,
                  'units_per_layer':256,
                  'num_nodes': 300,
                  'num_steps': 10,
                  'input_features': 3,
                  'output_features': 2,
                  'l2_reg': 0.00}


class FCLSTM(tf.keras.Model):

    def __init__(self, params=default_params):
        super().__init__()
        """
        Expected input: (batch, num_nodes, num_steps, input_features)
        """

        self.num_layers = params['num_layers']

        if type(params['units_per_layer']) is int:
            self.units_per_layer = [params['units_per_layer'] for _ in range(self.num_layers)]
        else:
            self.units_per_layer = params['units_per_layer']
            assert len(self.units_per_layer)==self.num_layers

        self.input_features = params['input_features']
        self.num_nodes = params['num_nodes']
        self.num_steps = params['num_steps']
        self.output_features = params['output_features']
        self.l2_reg = params['l2_reg']

        self.permuter_in = tf.keras.layers.Permute((2,1,3))
        self.reshaper_in = tf.keras.layers.Reshape((self.num_steps, -1))

        self.hidden_layers = []

        for i, units in enumerate(self.units_per_layer):

            return_sequences = True

            if i==len(self.units_per_layer)-1:
                return_sequences = True

            self.hidden_layers.append(tf.keras.layers.LSTM(units, activation='relu', return_sequences=return_sequences,
                                                           kernel_regularizer=tf.keras.regularizers.L2(self.l2_reg)))

        self.final_layer = tf.keras.layers.Dense(self.output_features*self.num_nodes, activation='linear')
        self.reshaper_out = tf.keras.layers.Reshape((self.num_steps, self.num_nodes, self.output_features))
        self.permuter_out = tf.keras.layers.Permute((2,1,3))

    def call(self, x):

        out = self.permuter_in(x)
        out = self.reshaper_in(out)

        for i in range(self.num_layers):
            out = self.hidden_layers[i](out)

        out = self.final_layer(out)
        out = self.reshaper_out(out)
        out = self.permuter_out(out)

        return out
