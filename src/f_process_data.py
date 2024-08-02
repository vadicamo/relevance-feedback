import pandas as pd
import numpy as np


def process_data(csv_mapping_path, csv_query_judgment_path, threshold_1=200, threshold_0=1400):
    '''
    This function reads the CSV files and processes the data to create ..
    
    '''
    # Read CSV files
    df_visione_mapping = pd.read_csv(csv_mapping_path, dtype={'column_name': str})
    df_query_judgment = pd.read_csv(csv_query_judgment_path)
    
    # Count judgment values for each query
    count_df = df_visione_mapping.groupby(['query'])['judgment'].value_counts().unstack(fill_value=0).reset_index()
    df_complete = pd.merge(df_visione_mapping, count_df, on='query', suffixes=('', '_count'))
    
    # Add the sum of values 0 and 1
    df_complete['sum_0_1'] = df_complete[1] + df_complete[0]
    df_complete_unique = df_complete.drop_duplicates(subset=['query'])
    df_complete_unique = df_complete_unique[['query', 'sum_0_1', 0, 1, -1]]
    df_complete_unique_sorted = df_complete_unique.sort_values(by=[1], ascending=True)
    
    # Filter rows with 1 > threshold_1 and 0 > threshold_0
    df_complete_unique_sorted = df_complete_unique_sorted[df_complete_unique_sorted[1] > threshold_1]
    df_complete_unique_sorted = df_complete_unique_sorted[df_complete_unique_sorted[0] > threshold_0]

    # Select only queries that meet the conditions
    queries = df_complete_unique_sorted['query'].values
    df_visione_mapping_query = df_visione_mapping[df_visione_mapping['query'].isin(queries)]
    
    # Create dictionary of shots with corresponding queries and judgments
    shot_labels_query = {}
    for index, row in df_visione_mapping_query.iterrows():
        shot_labels_query[row['visioneShotID'], row['query']] = row['judgment']
    
 
    return df_visione_mapping, df_query_judgment, df_complete, df_complete_unique_sorted, df_visione_mapping_query, shot_labels_query,queries



def get_gt_dict(df):
    '''
    df: DataFrame with the dataset one column for each image and the last row with the GT relevance
    '''
    # Function to get a dictionary from a DataFrame
    groundtruth_dic = {}
    for col in df.columns:
         groundtruth_dic[col] = df.iloc[-1][col]
    return groundtruth_dic



