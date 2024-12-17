#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Machine Learning: Modelo Linear com ElasticNet para prever a qualidade do vinho
===============================================================================
Este script serve para prever a qualidade de vinhos e para isso, utiliza o modelo ElasticNet.




The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
"""
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import argparse

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Definição de cores ANSI:
GREEN = "\033[92m"
RED = "\033[91m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"




def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    logging.info(f"{GREEN}Para executar os Hiperparâmetros, use o comando: python modeloLinear_ElasticNet.py --alpha 0.6 --l1_ratio 0.7{RESET}")
    # Read the wine-quality csv file from the URL:
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logging.exception(
            "Não foi possível baixar os dados de treinamento & teste CSV, verifique sua conexão com a internet. Error: %s", e
        )

    # Visualizando os dados (csv_url) em um DataFrame:
    logging.info(f"{RED}Visualizando os dados de vinhos em um DataFrame{RESET}")
    df = pd.read_csv(csv_url, sep=";")
    print(df.head())
    print(df.shape)

    # Dividimos os dados em conjuntos de treinamento e teste. (0,75, 0,25) divisão:
    train, test = train_test_split(data, test_size=0.25, random_state=42)

    # A coluna predita é "quality", que é um escalar de [3, 9]:
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Configurar o parser de argumentos:
    parser = argparse.ArgumentParser(description='Script para processar alpha e l1_ratio')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Valor de alpha (padrão: 0.5)')
    parser.add_argument('--l1_ratio', type=float, default=0.5,
                        help='Valor de l1_ratio (padrão: 0.5)')

    # Parsear os argumentos:
    args = parser.parse_args()

    # Usar os valores:
    print(f"Alpha: {args.alpha}, L1 Ratio: {args.l1_ratio}")



    with mlflow.start_run():
        lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(args.alpha, args.l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("l1_ratio", args.l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        
        # For remote server only (Dagshub):
        remote_server_uri = "https://dagshub.com/EddyGiusepe/MLflow_basic.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)


        # For remote server only (AWS)
        # remote_server_uri = "http://ec2-54-147-36-34.compute-1.amazonaws.com:5000/"
        # mlflow.set_tracking_uri(remote_server_uri)



        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")