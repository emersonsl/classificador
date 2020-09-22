import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier


#Leitura do arquivo

file = pd.read_csv('breast-cancer.csv',index_col='id_number' ,names=["id_number", "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape", "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei" ,"bland_chromatin", "normal_nucleoli", "mitoses", "class"])

#transformando a saída p/ binária: 0 para benigno e 1 para maligno

file['class'] = file['class'].replace(2,0)

file['class'] = file['class'].replace(4,1)

#removendo interrogações do dataset

file['bare_nuclei'] = file['bare_nuclei'].replace('?',randint(1,10))

#exibição do DataSet apos o tratamento
print("DATA SET\n")
print(file)
print("tamanho do DataSet: "+str(file.shape))

#separando tabela de alvos

out = file['class']

#separando a tabela de entrada, removendo valores de alvos

inp = file.drop(columns=['class'])

#separando conjunto de treino do conjunto de teste, proporção 70, 30, respectivamente

inp_train, inp_test, out_train, out_test = train_test_split(inp, out, test_size=0.3)

print('tamanho do conjunto entrada de treino: '+str(inp_train.shape))
print('tamanho do conjunto alvo de treino: '+str(out_train.shape))

print('tamanho do conjunto entrada de test: '+str(inp_test.shape))
print('tamanho do conjunto saída de test: '+str(out_test.shape)+'\n')


#treinando o conjunto de dados

clf = MLPClassifier(activation='logistic', max_iter=750, hidden_layer_sizes=245)
clf.fit(inp_train, out_train)

#testando o conjunto de dados

print("acuracia: "+str(clf.score(inp_test, out_test)))
