#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:23:08 2023

@author: Equipo DSA
"""

# Importe MLflow, ketas y tensorflow
import mlflow
import mlflow.keras
import keras
import tensorflow as tf
import tensorflow.keras as tk
from keras import models
from keras import layers


# Usaremos argparse para pasarle argumentos a las funciones de entrenamiento
import argparse

parser = argparse.ArgumentParser(description='Entrenamiento de una red feed-forward para el problema de clasificación con datos MNIST en TensorFlow/Keras')
parser.add_argument('--batch_size', '-b', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=5)
parser.add_argument('--learning_rate', '-l', type=float, default=0.05)
parser.add_argument('--num_hidden_units', '-n', type=int, default=512)
parser.add_argument('--num_hidden_layers', '-N', type=int, default=1)
parser.add_argument('--dropout', '-d', type=float, default=0.25)
parser.add_argument('--momentum', '-m', type=float, default=0.85)


args = parser.parse_args([])

# Usaremos esta función para definir Descenso de Gradiente Estocástico como optimizador
def get_optimizer():
    """
    :return: Keras optimizer
    """
    optimizer = keras.optimizers.SGD(learning_rate=args.learning_rate,momentum=args.momentum, nesterov=True)
    return optimizer


# Obtenemos el dataset MNIST 
mnist = tk.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalizamos los datos para que estén entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Esta función define una corrida del modelo, con entrenamiento y 
# registro en MLflow
def run_mlflow(run_name="MLflow CE MNIST"):
    # Iniciamos una corrida de MLflow
    mlflow.start_run(run_name=run_name)
    run = mlflow.active_run()
    # MLflow asigna un ID al experimento y a la corrida
    experimentID = run.info.experiment_id
    runID = run.info.run_uuid
    # reistro automáticos de las métricas de keras
    mlflow.keras.autolog()
    model = models.Sequential()
    #
    # La primera capa de la red transforma las imágenes de un arreglo 2d (28x28 pixels),
    # en un arreglo 1d de 28 * 28 = 784 pixels.
    model.add(layers.Flatten(input_shape=x_train[0].shape))
    # Agregamos capas ocultas a la red
    # en los argumentos: --num_hidden_layers o -N 
    for n in range(0, args.num_hidden_layers):
        # agregamos una capa densa (completamente conectada) con función de activación relu
        model.add(layers.Dense(args.num_hidden_units, activation=tf.nn.relu))
        # agregamos dropout como método de regularización para aleatoriamente descartar una capa
        # si los gradientes son muy pequeños
        model.add(layers.Dropout(args.dropout))
        # capa final con 10 nodos de salida y activación softmax 
        model.add(layers.Dense(10, activation=tf.nn.softmax))
        # Use Scholastic Gradient Descent (SGD) or Adadelta
        # https://keras.io/optimizers/
        optimizer = get_optimizer()

    # compilamos el modelo y definimos la función de pérdida  
    # otras funciones de pérdida comunes para problemas de clasificación
    # 1. sparse_categorical_crossentropy
    # 2. binary_crossentropy
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    # entrenamos el modelo
    print("-" * 100)
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    # evaluamos el modelo
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    mlflow.end_run(status='FINISHED')
    return (experimentID, runID)


# corrida con parámetros diferentes a los por defecto
args = parser.parse_args(["--batch_size", '256', '--epochs', '8'])
(experimentID, runID) = run_mlflow()
print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
print(tf.__version__)
print("-" * 100)