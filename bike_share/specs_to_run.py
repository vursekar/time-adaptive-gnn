import numpy as np

num_time_dims = 2

base_specs = {'save_model': True,
              'save_model_history': True,
              'batch_size': 64,
              'max_epochs': 50,
              'patience': 10,
              'loss_type': 'mae',
              'learning_rate': 1e-3}


models_to_run = {'LinearRegressor': [{'model_name': 'LinearRegressor', 'model_type': 'LinearRegressor'}|base_specs,
                                     {'num_nodes': 300,
                                      'num_steps': 10,
                                      'input_features': 3,
                                      'output_features': 2,
                                      'l2_reg': 0.00}],
                 
                 'FNN_noreg': [{'model_name': 'FNN_noreg', 'model_type': 'FNN'}|base_specs,
                                     {'num_layers': 2,
                                      'units_per_layer':256,
                                      'num_nodes': 300,
                                      'num_steps': 10,
                                      'input_features': 3,
                                      'output_features': 2,
                                      'l2_reg': 0.00}],
                 
                 'FNN_withreg': [{'model_name': 'FNN_woreg', 'model_type': 'FNN'}|base_specs,
                                     {'num_layers': 2,
                                      'units_per_layer':256,
                                      'num_nodes': 300,
                                      'num_steps': 10,
                                      'input_features': 3,
                                      'output_features': 2,
                                      'l2_reg': 0.001}],
                 
                 'FCLSTM': [{'model_name': 'FCLSTM', 'model_type': 'FCLSTM'}|base_specs,
                                     {'num_layers': 2,
                                      'units_per_layer':16,
                                      'num_nodes': 300,
                                      'num_steps': 10,
                                      'input_features': 3,
                                      'output_features': 2,
                                      'l2_reg': 0.001}],
                 
                 'STGCN_fixedA': [{'model_name': 'STGCN_fixedA', 'model_type': 'STGCN'}|base_specs,
                                     {'num_nodes': 300,
                                      'num_steps': 10,
                                      'input_features': 3,
                                      'output_features': 2,
                                      'num_time_dims': num_time_dims,
                                      'num_blocks': 2,
                                      'block_units': [[64,16,64], [32,16,32]],
                                      'block_kernels': [3,3],
                                      'A': np.eye(300),
                                      'emb_dim': 32,
                                      'add_identity': False,
                                      'time_modulation': False,
                                      'time_mod_layers': 0,
                                      'time_mod_activations':['relu','tanh'] ,
                                      'l2_reg': 0.00}],
                 
                 'STGCN_adaptiveA': [{'model_name': 'STGCN_adaptiveA', 'model_type': 'STGCN'}|base_specs,
                                     {'num_nodes': 300,
                                      'num_steps': 10,
                                      'input_features': 3,
                                      'output_features': 2,
                                      'num_time_dims': num_time_dims,
                                      'num_blocks': 2,
                                      'block_units': [[64,16,64], [32,16,32]],
                                      'block_kernels': [3,3],
                                      'A': None,
                                      'emb_dim': 32,
                                      'add_identity': True,
                                      'time_modulation': False,
                                      'time_mod_layers': 0,
                                      'time_mod_activations':['relu','tanh'] ,
                                      'l2_reg': 0.00}],
                 
                 'STGCN_timeadaptiveA': [{'model_name': 'STGCN_timeadaptiveA', 'model_type': 'STGCN'}|base_specs,
                                     {'num_nodes': 300,
                                      'num_steps': 10,
                                      'input_features': 3,
                                      'output_features': 2,
                                      'num_time_dims': num_time_dims,
                                      'num_blocks': 2,
                                      'block_units': [[64,16,64], [32,16,32]],
                                      'block_kernels': [3,3],
                                      'A': None,
                                      'emb_dim': 32,
                                      'add_identity': True,
                                      'time_modulation': True,
                                      'time_mod_layers': 1,
                                      'time_mod_activations':['relu','tanh'] ,
                                      'l2_reg': 0.00}],
                 
                 'GWN_adaptive': [{'model_name': 'GWN_adaptive', 'model_type': 'GraphWaveNet'}|base_specs,
                                     {'num_nodes': 300,
                                      'num_steps': 10,
                                      'input_features': 3,
                                      'output_features': 2,
                                      'num_time_dims': num_time_dims,
                                      'emb_dim': 32,
                                      'num_blocks': 4,
                                      'num_layers': 2,
                                      'res_channels': 32,
                                      'skip_channels': 32,
                                      'dilation_channels': 32,
                                      'end_channels': 32,
                                      'time_modulation': False,
                                      'time_mod_layers': 0,
                                      'time_mod_activations': ['relu','tanh'],
                                      'kernel_size': 2,
                                      'use_gc': True,
                                      'add_identity': True
                                      }],
                 
                 'GWN_timeadaptive': [{'model_name': 'GWN_timeadaptive', 'model_type': 'GraphWaveNet'}|base_specs,
                                     {'num_nodes': 300,
                                      'num_steps': 10,
                                      'input_features': 3,
                                      'output_features': 2,
                                      'num_time_dims': num_time_dims,
                                      'emb_dim': 32,
                                      'num_blocks': 4,
                                      'num_layers': 2,
                                      'res_channels': 32,
                                      'skip_channels': 32,
                                      'dilation_channels': 32,
                                      'end_channels': 32,
                                      'time_modulation': True,
                                      'time_mod_layers': 1,
                                      'time_mod_activations': ['relu','tanh'],
                                      'kernel_size': 2,
                                      'use_gc': True,
                                      'add_identity': True
                                      }],
            
                
                }
