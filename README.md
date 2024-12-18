# <h1 align="center">MLflow basic</h1>


<font color="pink">Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro</font>


Aqui usamos MLflow para rastreiar nossos experimentos e aprender a conectar `DagsHub` com o `GitHub`.

 
 O script `modeloLinear_ElasticNet.py` já está configurado para usar o `DagsHub` para o MLflow tracking. Mas você pode se conectar ao MLflow executando na linha de comando o seguinte:

 * Para DagsHub:

MLFLOW_TRACKING_URI=https://dagshub.com/EddyGiusepe/MLflow_basic.mlflow

MLFLOW_TRACKING_USERNAME=EddyGiusepe

MLFLOW_TRACKING_PASSWORD=134254756867879879123hsdfgsdfs921304

python modeloLinear_ElasticNet.py

Tendo isso você pode executar no terminal o seguinte:

```

export MLFLOW_TRACKING_URI=https://dagshub.com/EddyGiusepe/MLflow_basic.mlflow

export MLFLOW_TRACKING_USERNAME=EddyGiusepe 

export MLFLOW_TRACKING_PASSWORD=134254756867879879123hsdfgsdfs921304

```






Thank God!