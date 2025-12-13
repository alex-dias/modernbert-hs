import os
import numpy as np
import pandas as pd
import logging
import sys

def main():

    for file in os.listdir('select_knn_complete_2'):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join('select_knn_complete_2', file))
            
            df['density_ratio'] = np.exp(np.log(df['density_russian']) - np.log(df['density_all'])) 
            
            df.to_csv(os.path.join('select_knn_complete_2', file), index=False)

if __name__ == '__main__':
    main()
