import tensorflow as tf
import numpy as np
import pickle

import dflat.neural_optical_layer.core.arch_DNN as MLP_models
from sklearn.utils import shuffle

def run_training_neural_model(model, epochs, miniEpoch=1000, batch_size=None, lr=1e-3, verbose=True, train=True):
    model.customLoadCheckpoint()
    inputData, outputData = model.returnLibraryAsTrainingData()
    xtrain, ytrain = shuffle(inputData, outputData, random_state=13)
    if batch_size is None:
        batch_size = xtrain.shape[0]
    model(xtrain[0:1, :])
    model.summary()
    if train:
        splitNumberSessions = np.ceil(epochs / miniEpoch).astype(int)
        if tf.config.list_physical_devices('GPU'):
            device_name = "/gpu:0"
        else:
            device_name = "/cpu:0"
        with tf.device(device_name):
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer, loss='mse')
            for _ in range(splitNumberSessions):
                trackhistory = model.fit(
                    xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=miniEpoch,
                    verbose=verbose,
                    validation_split=0.10,
                    shuffle=True
                )
                model.customSaveCheckpoint(
                    trackhistory.history["loss"],
                    trackhistory.history["val_loss"],
                    verbose=False
                )
    return

if __name__ == "__main__":
    # Always train neural models with float32 because we do not need float64 and that is not standard
    # run_training_neural_model(model=MLP_models.MLP_Nanofins_Dense1024_U350_H600(dtype=tf.float32), epochs=10, miniEpoch=1000, batch_size=2**19, lr=1e-3)
     run_training_neural_model(model=MLP_models.MLP_Nanofins_Dense512_U350_H600(dtype=tf.float32), epochs=50000, miniEpoch=1000,batch_size=8, lr=1e-3, verbose=False)

    #run_training_neural_model(model=MLP_models.MLP_Nanocylinders_Dense256_U180_H600(dtype=tf.float32), epochs=50000, miniEpoch=1000, batch_size=500, lr=1e-3, verbose=False)

