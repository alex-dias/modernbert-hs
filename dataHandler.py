import pandas as pd
from datasets import Dataset, DatasetDict

# Create a DatasetDict from a DataFrame
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
    
# Transform a DataFrame into two lists to be used in MFT test cases
def createTestCaseList(df):
    labels = df['pred_label_notmasked'].tolist()
    labels = ["hate" if x == 1 else "no hate" for x in labels]
    dct = {'text': df['generation'].tolist(), 'label': labels}
    return dct
    
# Create a DatasetDict for a specific id_term
def toxigenDataset(id_term, test_size=0.2, test_case=False, test_case_size=0.05, random_state=42):
    file_name = 'toxigen_masked_pred_' + id_term + '.csv'
    df = pd.read_csv('masked_data\\' + file_name)
    
    if test_case:
        df_test_case = df.sample(frac=test_case_size)
        df = df.drop(df_test_case.index)
        return createDDFromDF(df, test_size), createTestCaseList(df_test_case)
    
    return createDDFromDF(df, test_size)
    
# Create a combined DatasetDict from multiple id_terms
def getMultiToxigenDataset(id_terms, test_size=0.2, is_random=False, random_seed=42, test_case=False, test_case_size=0.05):
    dfs = []
    
    for id_term in id_terms:
        file_name = 'toxigen_masked_pred_' + id_term + '.csv'
        dfs.append(pd.read_csv('masked_data\\' + file_name))
    dfFinal = pd.concat(dfs, ignore_index=True)
        
    if is_random:
        dfFinal = dfFinal.sample(frac=1, random_state=random_seed)
        
    if test_case:
        df_test_case = dfFinal.sample(frac=test_case_size)
        dfFinal = dfFinal.drop(df_test_case.index)
        return createDDFromDF(dfFinal, test_size), createTestCaseList(df_test_case)
        
    return createDDFromDF(dfFinal, test_size)