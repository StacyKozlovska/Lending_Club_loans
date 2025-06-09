import pandas as pd

def map_emp_length(X):
    emp_length_mapping = {'<1': 1, '1-3': 2, '4-6': 3, '7-9': 4, '10+': 5}
    X['emp_length_years'] = X['emp_length_years'].map(emp_length_mapping)
    return X[['emp_length_years']]