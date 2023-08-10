import numpy as np

weights = np.array([13000,15000,12000])
weights = np.array([max(weights)/x for x in weights])
print(weights)

min_value= np.min(weights)
max_value= np.max(weights)
norm_data = np.float64((weights-min_value)/(max_value-min_value)).tolist()
print(norm_data)