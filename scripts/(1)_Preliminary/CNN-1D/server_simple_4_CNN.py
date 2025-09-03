import flwr as fl
#import tensorflow as fl
import keras
from keras.models import *
from keras.layers import *
#NUM_CLIENTS = 2
from tab2img.converter import Tab2Img

#model = Tab2Img()
model = keras.Sequential()

#model.add(Conv2D(16, (3, 3), activation ='relu', padding ='same')(input_image))
model.add(Conv2D(16, (3, 3), activation ='relu', padding ='same')

model.add(MaxPooling2D((2, 2), padding ='same'))
model.add(Conv2D(8, (3, 3), activation ='relu', padding ='same'))
model.add(MaxPooling2D((2, 2), padding ='same'))
model.add(Conv2D(8, (3, 3), activation ='relu', padding ='same'))
model.add(MaxPooling2D((2, 2), padding ='same'))

# Building the decoder of the Auto-encoder
model.add(Conv2D(8, (3, 3), activation ='relu', padding ='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation ='relu', padding ='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation ='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation ='sigmoid', padding ='same'))

#params = model.get_parameters()
# Working: FedAvg(99.87), FedAdagrad (00.13), Fedyogi(99.87), FedAvgM(48.77), FedOpt(.0013), FedAdam (00.13) (TODO: FedProx, QFedAvg, FedOptim - same as FedOpt?)
# FedAvg: (McMahan et al., 2017)
# FedProxy: Li et al. (2020)
# QFedAvg: Li et al. (2019)
# Reddi et al. (2021)

strategy = fl.server.strategy.FedAvg(
    # fraction_fit=0.025,  # Train on 25 clients (each round)
    # fraction_evaluate=0.05,  # Evaluate on 50 clients (each round)
    # min_fit_clients=2,
    # min_evaluate_clients=2,
    # min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    # on_fit_config_fn=fit_config,
    #initial_parameters=fl.common.ndarrays_to_parameters(params)
)
while True:
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=2), strategy=strategy)
