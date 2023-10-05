import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix

from google.colab import files



def start(name_csv):
  dataset = pd.read_csv(name_csv+'.csv',encoding='utf-8')

  df = pd.DataFrame(dataset)

  colunas_numericas = list(get_dataframe_columns_type(df))
  X = dataset[colunas_numericas] # Features

  if hasattr(df, 'classe'):
        y = dataset.classe      # Target variable (classe)
  return X,y, get_columns_types(df)

def get_dataframe_columns_type(df) ->list:
  # Obtenha os tipos das colunas em uma lista e remova 'dtype' das strings
  tipos_numericos = get_columns_types(df)
  atributos_numericos = df.select_dtypes(include=tipos_numericos)
  #print(tipos_numericos)
  return atributos_numericos.columns

def get_columns_types(df):
  return [str(t).replace('dtype(\'', '').replace('\')', '') for t in df.dtypes.unique()]

def PCA_application(name_csv,X,y,n_comp_perc):
  ##pca_obj = PCA(n_components=4)
  pca_obj = PCA(n_components=n_comp_perc, whiten=True)
  pca_result = pca_obj.fit_transform(X)
  #print(pca_result.shape)
  columns = ["pca_"+str(i) for i in range(1,pca_result.shape[1]+1)]
  #print(columns)
  pca_dataset = pd.DataFrame(data = pca_result, columns=columns)

  final_data = pca_dataset.join(y)
  final_data.head()

  df = pd.DataFrame(final_data)
  df.to_csv(name_csv+'_PCA.csv', index=False)
  return df

def download_transformed_file(name_csv):
  files.download(name_csv,'.csv')

def accuracy_test(name_csv,tipos_numericos):
  data = pd.read_csv(name_csv+'.csv',encoding='utf-8')

  #split dataset in features and target variable
  atributos_numericos = data.select_dtypes(include=tipos_numericos)

  colunas_numericas = list(atributos_numericos.columns)

  X = data[colunas_numericas] # Features
  y = data.classe # Target variable

  X.drop('classe',inplace=True,axis=1)#dropei a classe porque ela estava entrando no cálculo pelo y também, já que era int64 tbm, e não podia!!

  X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

  # Create KNeighborsClassifier object
  knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
  knn.fit(X_train_70, y_train_70)

  #Predict the response for test dataset
  y_pred = knn.predict(X_test_30)

  # Model Accuracy, how often is the classifier correct?
  acuracia = metrics.accuracy_score(y_test_30, y_pred)
  #print('Accuracy of ',name_csv,': %.3f' % acuracia)

  # Matriz de confusão p/ 30%
  confusion_matrix(y_test_30, y_pred)
  return acuracia

def LDA_application(name_csv) -> str:
  lda_obj = LinearDiscriminantAnalysis(n_components=1)
  lda_result = lda_obj.fit(X, y).transform(X)
  ##pca_result = pca_obj.fit_transform(X)

  #print(lda_result.shape)

  lda_dataset = pd.DataFrame(data = lda_result, columns = ['lda_01'])
  ## Juntando o atributo classe
  final_data = lda_dataset.join(y)

  # Visualização dos dados normalizados
  final_data.head()

  # Salvando Pessoa.csv transformado
  df = pd.DataFrame(final_data)
  df.to_csv(name_csv,'_LDA.csv', index=False)
  return (name_csv,'_LDA.csv')


def veryfing_best_percentage(name_csv):
  X,y,column_types = start(name_csv)
  accuracy_test(name_csv,column_types)

  best_accuracy = 0.0
  percentage_at_best = 0.1
  for i in range(1,100):

    new_pca_dataframe = PCA_application(name_csv,X,y,i/100.0)
    column_types = get_columns_types(new_pca_dataframe)
    current_accuracy = accuracy_test(name_csv+"_PCA",column_types)

    if current_accuracy >= best_accuracy:
      best_accuracy = current_accuracy
      percentage_at_best =  i / 100.0
    
  print("The best accuracy was ",str(best_accuracy)," with ",str(percentage_at_best)," of the original dataset")
    



if __name__=='__main__':
 # X,y,column_types = start("Waveform")
  #new_pca_dataframe = PCA_application("Waveform",X,y,0.34)
 # accuracy_test("Waveform",column_types)
 # column_types = get_columns_types(new_pca_dataframe)
#  print(column_types)
 # accuracy_test("Waveform_PCA",column_types)

  veryfing_best_percentage("/../../data/HOG_Feature_Descriptor/HOG_Transform")
