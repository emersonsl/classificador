#Abertura do arquivo

import pandas as pd

arquivo = pd.read_csv('breast-cancer.csv', names=["id_number", "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape", "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei" ,"bland_chromatin", "normal_nucleoli", "mitoses", "class"])

print(arquivo)

