import tensorflow as tf

default_params = {'num_nodes': 300,
                  'num_steps': 10,
                  'input_features': 3,
                  'output_features': 2,
                  'l2_reg': 0.00}


class LinearRegressor(tf.keras.Model):

    def __init__(self, params=default_params):
        super().__init__()
        """
        Expected input: (batch, num_nodes, num_steps, input_features)
        """

        self.input_features = params['input_features']
        self.num_nodes = params['num_nodes']
        self.num_steps = params['num_steps']
        self.output_features = params['output_features']
        self.final_shape = (self.num_nodes, self.num_steps, self.output_features)
        self.l2_reg = params['l2_reg']

        self.reshaper_in = tf.keras.layers.Reshape((self.num_nodes, -1))
        self.final_layer = tf.keras.layers.Dense(self.output_features*self.num_steps, activation='linear',
                                                 kernel_regularizer=tf.keras.regularizers.L2(self.l2_reg))
        self.reshaper_out = tf.keras.layers.Reshape(self.final_shape)

    def call(self, x):

        out = self.reshaper_in(x)
        out = self.final_layer(out)
        out = self.reshaper_out(out)

        return out
