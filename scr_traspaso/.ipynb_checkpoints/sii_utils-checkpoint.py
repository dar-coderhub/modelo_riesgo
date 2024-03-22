import warnings
import pandas as pd
import yaml
import numpy as np
import string

#para test
def suppress_warnings(tipo = 'all'):
    """Suprime todas las advertencias en el código."""
    if tipo == 'all':
        warnings.filterwarnings("ignore")
    if tipo == 'pw':
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)




def _convert_types(_conv_dataset: pd.DataFrame, filename = "config_types.yml"):
    
    """
        Descripcion:
            - Retornar las estadisticas descriptivas de un dataframe entregado
            
        Parámetros:
            - _conv_dataset (dataframe): Dataframe que contine las variables para el resumen de estadistica descriptiva

        Retorno:
           - Retorna un frame el resumen de las variables entregadas
    """

    variables = []
    inputed_types = []
    if filename[-4:] == '.yml':
        print(f'Leyendo archivo .yml : {filename}')
        with open(filename, 'r') as stream:
            config = yaml.safe_load(stream)

        for variable, inputed_type in config["dtypes"].items():
            variables.append(variable)
            inputed_types.append(inputed_type)


    if filename[-4:] == '.csv':
        print(f'Leyendo archivo .csv : {filename}')
        df_type = pd.read_csv(filename, sep=';')
        for idx, row in df_type.iterrows():
            variables.append(row['Variable'])
            inputed_types.append(row['Tipo'])


    #Definimos funciones para convertir a int, ,float, date o category
    #def convert_to_integer(target, default_value = -1):
     #       try:
      #          return int(target)
       #     except:
        #        return int(default_value)
    
    def convert_to_float(target, default_value = np.nan):
            try:
                return float(target)
            except:
                return float(default_value)

    def convert_to_date(target, default_value = None):
            try:
                target = target.astype('str')
                if target.str.find('-').sum() > 0:
                    if target.str.find(':').sum() > 0:
                        target = pd.to_datetime(target, format='%d-%m-%Y %H:%M:%S')
                    else:
                        target = pd.to_datetime(target, format='%d-%m-%Y')
                elif target.str.find('/').sum() > 0:
                    if target.str.find(':').sum() > 0: 
                        target = pd.to_datetime(target, format='%d/%m/%Y %H:%M:%S')
                    else: 
                        target = pd.to_datetime(target, format='%d/%m/%Y')
                elif target.str.find(string.ascii_letters).unique() == -1:
                    if target.apply(lambda x: len(x)).max() <= 10:
                        target = pd.to_datetime(target, format='%Y%m%d')
                    else: 
                        target = target.apply(lambda x: x.replace('.0',''))
                        target = pd.to_datetime(target, format ='%Y%m%d%H%M%S')
                return target
            except: 
                return target

    def convert_to_object(target, default_value = np.nan):
            try: 
                return str(target)
            except:
                return str(default_value)
    
    def convert_to_category(target, default_value = np.nan):
            try: 
                return int(target)
            except:
                return default_value
    
    #Aplicamos la conversion de variables
    for variable, inputed_type in zip(variables,inputed_types):

            if inputed_type == "datetime":
                _conv_dataset[variable] = convert_to_date(_conv_dataset[variable])
            #elif inputed_type == "int":
             #   _conv_dataset[variable] = _conv_dataset[variable].apply(lambda x : convert_to_integer(x)).astype(inputed_type)
            elif inputed_type == "float":
                _conv_dataset[variable] = _conv_dataset[variable].apply(lambda x : convert_to_float(x)).astype(inputed_type)
            #elif inputed_type == "category":
             #   _conv_dataset[variable] = _conv_dataset[variable].apply(lambda x : convert_to_category(x)).astype(inputed_type)
            elif inputed_type == "object":
                #print(variable, inputed_type)
                #print(_conv_dataset[variable].unique())
                _conv_dataset[variable] = _conv_dataset[variable].apply(lambda x : convert_to_object(x)).astype(inputed_type)
            #display(_conv_dataset[variable].describe())

    #print(_conv_dataset.dtypes)
    print("Conversion de data finalizada")
    return _conv_dataset



import numpy as np
import pandas as pd

def reemplazar_nulos(series, valor_reemplazo, verbose = 0):
    """
    Descripción:
        - Reemplaza los valores nulos en una serie según el valor especificado.

    Parámetros:
        - series (pd.Series): La serie de pandas.
        - valor_reemplazo (str, int, float): Valor a utilizar para el reemplazo de nulos.

    Retorno:
       - new_series (pd.Series): Datos posterior al reemplazo de nulos.
    """
    if verbose > 0:
        print("Reemplazando valores nulos en columns {} por el valor {}".format(series.name, valor_reemplazo))
    new_series = series.fillna(valor_reemplazo)
    new_series = new_series.replace('nan', valor_reemplazo)
    new_series = new_series.replace('NAN', valor_reemplazo)
    new_series = new_series.replace('NaN', valor_reemplazo)
    return new_series



import yaml
import pandas as pd

def apply_null_handling_strategy(df, config, verbose = 0):
    # Aplica las estrategias para columnas específicas
    column_strategies = config.get("column_strategies", [])
    df = apply_column_strategies(df, column_strategies, verbose)

    # Aplica la estrategia general
    general_strategy = config.get("general_strategy", {})
    df = apply_general_strategy(df, general_strategy, verbose)
    return df


import foo
def apply_general_strategy(df, strategy_configs, verbose = 0):
    print(f'Aplicando estrategia general de imputacion de datos')
    df = df.copy()
    for strategy_config in strategy_configs:
        dataframe_strategy = strategy_config.get("dataframe_strategy", "")
        strategy           = strategy_config.get("strategy", "")
        custom_function    = strategy_config.get("custom_function")
        thresh             = strategy_config.get("thresh", "")
        valor_reemplazo    = strategy_config.get("valor_reemplazo")

        if valor_reemplazo == 'np.nan':
            valor_reemplazo = np.nan

        if verbose > 0: print(f'Estrategia {dataframe_strategy}:{strategy}:{valor_reemplazo}')

        if strategy == "drop_rows":
            if verbose > 0: print("\tdrop_rows")
            ori_rows = df.shape[1]
            thresh_ = (1-thresh)*int(df.shape[1])
            df.dropna(thresh =  thresh_, inplace=True)
            if verbose > 0: print(f'\tdrop_rows: registros eliminados {ori_rows-df.shape[1]} - thresh = {thresh_}')
        elif strategy == "drop_columns":
            if verbose > 0: print("\tdrop_columns")
            ori_cols = df.columns
            df.dropna(thresh = (1-thresh)*int(df.shape[1]), axis=1, inplace=True)
            del_cols = list(set(df.columns)-set(ori_cols))
            if verbose > 0: print(f'\tdrop_columns: atributos eliminados { len(del_cols)}')
            if verbose > 0: print('\t'+ " ".join(del_cols))
        elif strategy == "fill_mean":
            df.fillna(df.mean(), inplace=True)
        elif strategy == "fill_median":
            df.fillna(df.median(), inplace=True)
        elif strategy == "custom_function" and custom_function == 'reemplazar_nulos':
            if verbose > 0: print("\tcustom_function reemplazar_nulos")
            custom_function = globals()[custom_function]
            if dataframe_strategy == 'all':
                columns = df.columns
            if dataframe_strategy == 'object':
                columns = df.select_dtypes(include = ['object']).columns
            if dataframe_strategy == 'numeric':
                columns = df.select_dtypes(include = ['number']).columns
            for column_name in columns:
                df[column_name] = custom_function(df[column_name], valor_reemplazo, verbose )
        elif strategy == "custom_function" and custom_function and custom_function != 'reemplazar_nulos':
            custom_function = globals()[custom_function]
            df = custom_function(df)
    return df

import foo
def apply_column_strategies(df, column_strategies, verbose):
    print(f'Aplicando estrategia de imputacion de datos por columna')
    df = df.copy()
    for column_strategy in column_strategies:
        column_name     = column_strategy.get("column_name", "")
        strategy        = column_strategy.get("strategy", "")
        custom_function = column_strategy.get("custom_function")
        valor_reemplazo = column_strategy.get("valor_reemplazo")

        if valor_reemplazo == 'np.nan':
            valor_reemplazo = np.nan

        if verbose > 0:
            print(f'Estrategia {column_name}: {strategy}: {valor_reemplazo}')

        if strategy == "drop_column":
            df.drop(columns=column_name, inplace=True)
        elif strategy == "fill_mean":
            df[column_name].fillna(df[column_name].mean(), inplace=True)
        elif strategy == "fill_median":
            df[column_name].fillna(df[column_name].median(), inplace=True)
        elif strategy == "custom_function" and custom_function == 'reemplazar_nulos':
            custom_function = globals()[custom_function]
            df[column_name] = custom_function(df[column_name], valor_reemplazo, verbose )
        elif strategy == "custom_function" and custom_function and custom_function != 'reemplazar_nulos':
            custom_function = globals()[custom_function]
            df[column_name] = custom_function(df[column_name])
    return df






###################################################
### DESCRIPTIVO
###################################################

def Descriptivo_categorico(data,names_columns):
    
    """
        Descripcion:
            -  Genera un descriptivo univariado para cada una de las variables categoricas, tales como:
                    - Cantidad de registros por cada categoria 
                    - Cantidad de datos NaN 
                    - Porcentaje que representa la cantidad de los datos obtenidos anteriormente
            
        Parámetros:
            - data (pd.DataFrame): Dataframe que contiene las variables a obtener estas estadistica univariadas
            - names_columns (list): Lista con los nombre de cada una de las variables categóricas

        Retorno:
        
         (pd.DataFrame) Un dataframe con las estadisticas de cada variable
        """
    ## Generacion de analitica para variables categoricas

    frec = np.unique(data[[names_columns]].astype('str'), return_counts=True)
    dfinal = pd.DataFrame(frec).T.copy()
    dfinal['porcentaje'] = dfinal.iloc[:,1]/data.shape[0]
    observaciones=np.nan
    dfinal[observaciones]=np.nan
    dfinal.columns = ["categorías", "frecuencia", "porcentaje", "observaciones"]
    return dfinal


def Descriptivo_numerico (data, names_columns):
    
    """
        Descripcion:
            -  Genera un descriptivo univariado para cada una de las variables numéricas, tales como:
                    - Cantidad de ceros
                    - Cantidad de valores negativos
                    - Cantidad de datos NaN y su porcentaje respecto del total
                    - Valor mínimo
                    - Valor máximo
                    - Media de cada variable
                    - Percentil 5, 10, 25, 50, 75 y 90
            
        Parámetros:
            - data (pd.DataFrame): Dataframe que contiene las variables a obtener estas estadistica univariadas
            - names_columns (list): Lista con los nombre de cada una de las variables numéricas

        Retorno:
            - (pd.DataFrame) Un dataframe con las estadisticas de cada variable
    """
    
    # Descriptivo de las variables continuas
    array = data[names_columns]
    n_zeros     = (array==0).sum() # N° de zeros
    n_negativos = (array<0).sum() # N° de valores negativos
    n_missing   = (array.isna()).sum() # N° de valores missing
    p_missing   = n_missing/array.shape[0]
    array = array[~array.isna()] ## Eliminar nulos
    min = array.min()
    max = array.max()
    mean = array.mean()
    print(array)
    p5   =array.quantile(.05) # Valor del percentil al 5%
    p10 = np.percentile(array, 10)
    p25 = np.percentile(array, 25)
    p50 = np.percentile(array, 50)
    p75 = np.percentile(array, 75)
    p90 = np.percentile(array, 90)
    observaciones=np.nan
    df=pd.DataFrame([{"n_zeros":n_zeros, "n_negativos": n_negativos,
                      "n_missing":n_missing,"p_missing":p_missing,"min":min,"max":max, "mean":mean,
                      "p5":p5,"p10":p10,"p25":p25,"p50":p50,"p75":p75,"p90":p90,
                      "observaciones": observaciones}]).T
    df.columns=[names_columns]
    return df


def Descriptivo(data,filename):
    
    """
        Descripcion:
            -  Genera un descriptivo univariado para cada una de las variables, depediendo del tipo de variable, ya sea numerica o categorica, y lo guarda en un archivo excel.
               Este archivo Excel contiene una pagina relacionado a las varaibles numericas y paginas para cada una de las variables categoricas
            
        Parámetros:
            - data (pd.DataFrame): Dataframe del cual se obtendrán las estadisticas descriptivas

        Retorno:
            - (File) Archivo Excel con las estadísticas univariadas de cada variable según su tipo
    """
    
    # Esta funcion genera un descriptivo depediendo del tipo de variable, numerica o categorica, y lo guarda en un archivo excel.
    # Este archivo Excel contiene una pagina relacionado a las varaibles numericas y paginas para cada una de las variables categoricas

    with pd.ExcelWriter(f"../artefactos/Descriptivo/Descriptivo_{filename}.xlsx") as writer:
        names_column_num = data.loc[:, (data.dtypes == "number")].columns.copy()
        names_column_cat = data.loc[:, (data.dtypes == "object")].columns.copy()
        n = 0
        if (names_column_num.shape[0] >= 1):
            desc_num_list = []
            for names_column in names_column_num:
                desc_num = Descriptivo_numerico(data, names_column)
                desc_num_list.append(desc_num)
            final_desc_num = pd.concat(desc_num_list, axis=1).T
            final_desc_num.to_excel(writer, "var_numericas")
            n += 1
        
        if (names_column_cat.shape[0] >= 1):
            for names_column in names_column_cat:
                n = n + 1
                desc_txt = Descriptivo_categorico(data, names_column)
                desc_txt.to_excel(writer, names_column, index=False)

    print("Se ha creado un descriptivo de los datos")
        






def Transformacion_valoresnegativos(series):     
    """
        Descripcion:
            - Transforma la variable en 2 nuevas variables: 1) el valor absoluto  y 2) el signo

        Parámetros:
            - series (pd.Series):

        Retorno:
           - absolute_series(pd.Series): Magnitud de los datos
           - sign_series(pd.Series): Signo de los datos
    """
    # Extract sign and absolute values
    sign_series = series.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    absolute_series = series.abs()

    return absolute_series, sign_series



###########################################################################
### Split the dataframe into test and train data
###########################################################################
from sklearn.model_selection import train_test_split
def split_data(df,target):
    X = df.loc[:,:].drop(target, axis=1)
    y = df.loc[:,:][target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}
           }
    return data # genera una variable que contiene dos tipos de elementos




# Metricas

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate,train_test_split,KFold

## Models

from sklearn.linear_model import SGDClassifier
from sklearn import  linear_model, naive_bayes,svm
from sklearn import decomposition, ensemble

##Plots
import matplotlib.pyplot as plt

## save
from pathlib import Path
import pickle


def True_negative(y_true, y_pred):  return confusion_matrix(y_true, y_pred)[0, 0]
def False_positive(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def False_negative(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def True_positive(y_true, y_pred):  return confusion_matrix(y_true, y_pred)[1, 1]


from scipy.stats import ks_2samp

def ks_statistic(sample1, sample2):
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic for two samples.

    Parameters:
    - sample1: First sample
    - sample2: Second sample

    Returns:
    - ks_stat: KS statistic
    """
    ks_stat, _ = ks_2samp(sample1, sample2)
    return ks_stat.statistic



def Score_metrics():
    
    """
        Descripcion:
            - Creaccion de diccionario de metricas para modelos de clasificacion binario

        Parámetros:
    

        Retorno:
           -diccionario(dict): Diccionario de metricas
    """
    
          

    score_metrics = {'acc': accuracy_score,
               'balanced_accuracy': balanced_accuracy_score,
               'prec': precision_score,
               'recall': recall_score,
               'f1-score': f1_score,
               'tp': True_positive, 'tn': True_negative,
               'fp': False_positive, 'fn': False_negative,
               "roc_auc":roc_auc_score}

    return score_metrics

def Dict_models():
    
    """
        Descripcion:
            - Creaccion de diccionario de modelos

        Parámetros:
    

        Retorno:
           -diccionario(dict): Diccionario de los modelos a utilizar en el champion challenger
    """
    

    dict_models={

        'SVM':svm.SVC(kernel='linear',max_iter=1000),
        'SGDClassifier':SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3,   n_iter_no_change=10, early_stopping=True, n_jobs=-1 ),
        'Logistic':linear_model.LogisticRegression(max_iter=1000),
        'Random_forest':ensemble.RandomForestClassifier(n_estimators=101,bootstrap=True,min_impurity_decrease=1e-7,n_jobs=-1, random_state=42)

    }
    #,'SGDClassifier':SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3,   n_iter_no_change=10, early_stopping=True, n_jobs=-1 )
    #'Random_forest':ensemble.RandomForestClassifier(bootstrap=True,min_impurity_decrease=1e-7,n_jobs=-1, random_state=42)


    return dict_models

def Reporte(clf,x,y,name='classifier',name_feature='features',name_pre='preporcess' ,cv=3, dict_scoring=None, fit_params=None):
    
    """
        Descripcion:
            - Crea un reporte de comparacion de diferentes modelos de clasificacion

        Parámetros:
        
           - clf  (sklearn.model): Modelo para el entrenamiento
           -x(pd.DataFrame): Variables para el entrenamiento
           -y(pd.DataFrame): Vector objetivo
           -name(str): Nombre del modelo
           -name_feature(str): Nombre de la caracteristica
           -name_pre(str): Nombre del pre-procesamiento 
           -cv=3(float/sklearn.model_select (Kfold)): Metodo para la creacion del train and test datasets.  
           -dict_scoring(dict): diccionario de metricas a evaluar de un modelo de clasificacion binario
           -fit_params: None
    

        Retorno:
           - (pd.DataFrame): Resumen del entrenamiento del modelo
    """
    
    
    
            
    #print(dict_scoring)
    
    if dict_scoring!=None:
        score = dict_scoring.copy()
        for i in score.keys():
            score[i] = make_scorer(score[i])
    
    scores = cross_validate(clf, x, y, scoring=score, cv=cv, return_train_score=True, n_jobs=-1,  fit_params=fit_params)
    
    index = []
    value = []
    index.append("Modelo")
    value.append(name)
    index.append("Feature")
    value.append(name_feature)
    index.append('Preprocessing')
    value.append(name_pre)
    for i in scores:
        if i == "estimator":
            continue
        for j in enumerate(scores[i]):
            index.append(i+"_cv"+str(j[0]+1))
            value.append(j[1])
        #if any(x in i for x in scoring.keys()):
        
        index.append(i+"_mean")
        value.append(np.mean(scores[i]))
        index.append(i+"_std")
        value.append(np.std(scores[i]))
        
    return pd.DataFrame(data=value, index=index).T,clf

def Train_model(data, seleccionadas,model,path_artefactos,modelname):
    
    """
        Descripcion:
            - Entrenamiento de un modelo de clasificacion binario

        Parámetros:
        
           -X(pd.DataFrame): Variables para el entrenamiento
          -y(pd.DataFrame): Vector objetivo
          -model  (sklearn.model): Modelo para el entrenamiento
          -path_artefactos(str):Path para guardar el modelo en la carperta de artefactos 
    

        Retorno:
             - model: Modelo entrenado
           -data_test(tupla): (y_test,y_predict) 
    """
    
    
    #cfg=import_config()
    import yaml

    with open('../config_file.yml', 'r') as file:
        cfg = yaml.safe_load(file)



    model.fit(data['train']['X'][seleccionadas],data['train']['y'])
    y_predict=model.predict(data['test']['X'][seleccionadas])

    score=model.score(data['test']['X'][seleccionadas],data['test']['y'])

    print('Modelo {} con un Score de {:.2f}'.format(modelname, score))   
    path_save=Path(cfg['path']['Home'],cfg['path']['Artefactos'],path_artefactos,'Modelo',f'modelo_{modelname}.pkl')
    Save_model(model,path_save)

    return model



def Matriz_confusion(y_actual,y_pred):
    
    """
        Descripcion:
            - Computa la matriz de confusion

        Parámetros:
        
        -y_actual(pd.series/np.array): Valor real del vector objetivo
          -y_pred(pd.series/np.array): valor predicho del vector objetivo
        Retorno:
             - plt.plot: Matriz de confusion
    """
    
    
    cfm = confusion_matrix(y_actual, y_pred)
    
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cfm, display_labels = [True, False])
    cm_display = cm_display.plot(cmap=plt.cm.Blues,values_format='g')
    plt.show()
    
    
def Matriz_confusion_per(y_actual,y_pred):
    
    """
        Descripcion:
            - Computa la matriz de confusion

        Parámetros:
        
        -y_actual(pd.series/np.array): Valor real del vector objetivo
          -y_pred(pd.series/np.array): valor predicho del vector objetivo
        Retorno:
             - plt.plot: Matriz de confusion en porcentaje
    """
    
    
    
    cfm = confusion_matrix(y_actual, y_pred)
    cmn = cfm.astype('float') /cfm.sum(axis=1)[:, np.newaxis]
    cm_display =  ConfusionMatrixDisplay(confusion_matrix = cmn, display_labels = [True, False])
    cm_display = cm_display.plot(cmap=plt.cm.Blues,values_format='.2g')

    plt.show()



def Save_model(model,path_model):
    
    
    """
        Descripcion:
            - Guarda el modelo

        Parámetros:
       
          --model  (sklearn.model): Modelo para el entrenamiento
          -path_model(str): Path en donde se guardara el modelo como artefacto
          
    """
    


    pickle.dump(model, open(path_model, 'wb'))

    print(f'Modelo Guardado en {path_model}')
    return 0



###########################################################################
### Metodo de calculo de indicadores de desempeño
###########################################################################
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

def evaluate_classification(y_true,y_pred): 
    auc        = roc_auc_score(y_true, y_pred)
    ks_stat, _ = ks_2samp(y_pred[y_true == 0], y_pred[y_true == 1])
    if auc < 0.5: auc = 1 - auc
    return auc, ks_stat
    

###########################################################################
### Obtener estadisticas de los atributos
###########################################################################
def GetFeaturesStatistics(data, seleccionadas):
    predVar = []
    for f in seleccionadas:
      try:
        feature = data['train']['X'][f]
        metricas = evaluate_classification(data['train']['y'], feature)
        predVar.append([f, metricas[0],metricas[1]])
      except Exception as E:
        pass
    predVar = pd.DataFrame(predVar, columns = ['Feature','AUC','KS'])
    seleccionadas = list(predVar['Feature'].values)
    return predVar, seleccionadas

###########################################################################
### Seleccion por correlacion GLOBAL
###########################################################################
def SelCorrGlobal(data, predVar,  corThr = 0.7):
    seleccionadas = []
    predVarFam = predVar.sort_values('AUC',ascending = False)
    variables_candidatas    = predVarFam['Feature']
    variables_revisadas     = []
    variables_seleccionadas = []
    corMtx = data.loc[:,variables_candidatas].corr()
    for f in variables_candidatas:
        if f not in variables_revisadas:
            variables_revisadas.append(f)
            variables_seleccionadas.append(f)
            corTemp = list(corMtx.index[(np.abs(corMtx[f]) >= corThr)])
            for fx in corTemp:
                if fx != f and fx not in variables_revisadas:
                    variables_revisadas.append(fx)
    seleccionadas = seleccionadas + variables_seleccionadas
    predVar = predVar.loc[predVar['Feature'].isin(seleccionadas),:]
    return predVar, seleccionadas





import os
def mkdir_if_not_exists(directory_path):
    """
    Create a directory if it does not exist.

    Parameters:
    - directory_path: Path of the directory to be created.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

