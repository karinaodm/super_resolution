import os
import numpy as np

data_settings = {'width': 64,
                 'height': 64,
                 'n_channels': 3,
                 'real_label': 1,
                 'fake_label': 0,
                 'batch_size': 16,
                 'seed': 1298,
                 'path2data': os.path.join('..', 'data', 'COCO'),
                 'workers': 0,
                 }

training_settings = {'working_dir': os.path.join('..', 'data', 'models'),
                     'epochs': 10,
                     'learning_rate': 0.001,
                     'n_gpu': 1,
                     }

models_settings = {'ndf': 64,  # Size of feature maps in discriminator
                  }