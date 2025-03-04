import pandas as pd
from datasets import Dataset, DatasetDict

def createDDFromDF(df, test_size=0.2):
    df = df[['generation', 'pred_label_notmasked']]
    df = df.rename(columns={'generation': 'text', 'pred_label_notmasked': 'label'})
    
    df_hate = df[df['label'] == 1]
    df_no_hate = df[df['label'] == 0]
    
    df_hate_train = df_hate.sample(frac=1-test_size)
    df_hate_test = df_hate.drop(df_hate_train.index)
    
    df_no_hate_train = df_no_hate.sample(frac=1-test_size)
    df_no_hate_test = df_no_hate.drop(df_no_hate_train.index)
    
    df_train = pd.concat([df_hate_train, df_no_hate_train])
    df_test = pd.concat([df_hate_test, df_no_hate_test])
    
    df_train = df_train[['text', 'label']]
    df_test = df_test[['text', 'label']]
    
    df_train_dataset = Dataset.from_pandas(df_train).remove_columns(['__index_level_0__'])
    df_test_dataset = Dataset.from_pandas(df_test).remove_columns(['__index_level_0__'])
    
    return DatasetDict({
            "train": df_train_dataset,
            "test": df_test_dataset
            })
    
    
def toxigenDataset(id_term, test_size=0.2):
    file_name = 'toxigen_masked_pred_' + id_term + '.csv'
    df = pd.read_csv('masked_data\\' + file_name)
    
    return createDDFromDF(df, test_size)

    
def getMultiToxigenDataset(id_terms, test_size=0.2, is_random=False, random_seed=42):
    dfs = []
    
    for id_term in id_terms:
        file_name = 'toxigen_masked_pred_' + id_term + '.csv'
        dfs.append(pd.read_csv('masked_data\\' + file_name))
    dfFinal = pd.concat(dfs, ignore_index=True)
        
    if is_random:
        dfFinal = dfFinal.sample(frac=1, random_state=random_seed)
        
    return createDDFromDF(dfFinal, test_size)