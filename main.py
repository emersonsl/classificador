import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier

#Leitura do arquivo

file = pd.read_csv('breast-cancer.csv',index_col='id_number' ,names=["id_number", "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape", "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei" ,"bland_chromatin", "normal_nucleoli", "mitoses", "class"])

#transformando a saída p/ binária: 0 para benigno e 1 para maligno

file['class'] = file['class'].replace(2,0)

file['class'] = file['class'].replace(4,1)

#removendo dados com parametros equivocados

file_remove = file.loc[file['bare_nuclei']=='?']

file = file.drop(file_remove.index)

#separando tabela de alvos

out = file['class']

#separando a tabela de entrada, removendo o id e os alvos

inp = file.drop(columns=['class'])

#separando conjunto de treino do conjunto de teste, proporção 70, 30, respectivamente

inp_train, inp_test, out_train, out_test = train_test_split(inp, out, test_size=0.3)

#treinando o conjunto de dados

clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(inp_train, out_train)

print(clf.score(inp_test, out_test))