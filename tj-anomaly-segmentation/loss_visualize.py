import matplotlib.pyplot as plt
import numpy as np
import csv

with open('C:/Users/11090/Desktop/resynthesis/cond_transformer_image256_neim8192/252epoch/testtube/version_0/metrics.csv') as f:
    f_csv = csv.reader(f)
    print(f_csv)
    for row in f_csv:
        print(row)
