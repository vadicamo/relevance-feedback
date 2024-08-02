from datetime import datetime
import hashlib
import json
import pandas as pd
import numpy as np
from f_polyquery_msed_logscale import polyquery_msed_logscale
from f_rocchio import rocchio 
from f_pichunter_star import pichunter
from f_polyadic import polyquery
from f_svm import svm
from f_polyquery_msed_logscale import polyquery_msed_logscale
from f_process_data import *
from tqdm import tqdm
import os
import inspect



def data_loading():
    '''
    Load the data and create the main dataframes
    Parameters:
    ----------
    None
    Returns:
    ----------
    df_visione_mapping: pd.DataFrame
        dataframe with the visione mapping data ??
    df_query_judgment: pd.DataFrame
        dataframe with the query judgment data??
    df_complete: pd.DataFrame
        dataframe with the complete data ??
    df_complete_unique_sorted: pd.DataFrame
        dataframe with the complete unique sorted data ??
    df_visione_mapping_query: pd.DataFrame
        dataframe with the visione mapping query data ??
    shot_labels_query: pd.DataFrame
        dataframe with the shot labels query data ??
    queries: pd.DataFrame
        dataframe with the queries data ?? 
    '''
    df_visione_mapping, df_query_judgment, df_complete, df_complete_unique_sorted, df_visione_mapping_query, shot_labels_query,queries  = process_data('/home/francescascotti/data/avsGT/avs_gt_visione_mapping.csv', '/home/francescascotti/data/avsGT/avs_gt_visione_mapping_query_judgment.csv', threshold_1=200, threshold_0=1400)
    return df_visione_mapping, df_query_judgment, df_complete, df_complete_unique_sorted, df_visione_mapping_query, shot_labels_query,queries

def save_results(exp_params, output_filename,action_dic, num_positives_at_iter,map_results,ndcg_results, recall_results):
    """
    Save the results of the experiment in a jsonl file.
    """

    with open(output_filename, 'a') as f:
        records = []
        for i, (action_step, n_pos_at_iter, all_actions, map_val, ndcg_val, recall_val) in enumerate(zip(action_dic.keys(), num_positives_at_iter,action_dic.values(), map_results, ndcg_results, recall_results)):
            results_record={
                            'iteration': i,
                            'n_pos_at_iter': n_pos_at_iter, #number of positive images in the display at the iteration
                            'recall': recall_val,
                            'ndcg': ndcg_val,
                            'map': map_val,
                            'action_step': action_step, #should be the same as iteration 
                            'n_actions': len(all_actions), #total number of actions in the iteration
                            'n_pos_action':len([act for act, act_type  in all_actions if act_type==1]), #number of positive actions in the iteration
                            'n_neg_action':len([act for act, act_type  in all_actions if act_type==0]), #number of negative actions in the iteration
                            'actions': all_actions#list of all the actions 
            }
           
            results_record.update(exp_params)
            records.append(results_record)
        #write the results in the jsonl output file
        for record in records:
             record_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in record.items()}
             f.write(json.dumps(record_serializable)+'\n')
        f.close()


def filter_arguments(func, kwargs):
    """
    Filter kwargs based on func's signature.
    """
    sig = inspect.signature(func)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return filtered_kwargs

def get_params(exp_params, method, fun_name,data_preprocessing, use_batch):
    new_params=exp_params.copy()
    new_params["method"]=method
    new_params["fun_name"]=fun_name
    new_params["data_preprocessing"]=data_preprocessing
    new_params["initial_query"]=None
    new_params["use_batch"]=use_batch
    exp_id=ged_id_for_dict(exp_params)
    new_params["exp_id"]=exp_id
    return new_params

def run_experiment(exp_params,output_filename, dataset_df):
    new_params=exp_params.copy()
    if exp_params["method"]=="Rocchio":
        filtered_params = filter_arguments(rocchio, new_params)# Filter out unnecessary arguments
        eval_res = rocchio(dataset_df, **filtered_params) 
        save_results(new_params, output_filename, **eval_res)
    elif exp_params["method"]=="Pichunter":
        filtered_params = filter_arguments(pichunter, new_params)
        eval_res = pichunter(dataset_df, **filtered_params)
        save_results(new_params,output_filename, **eval_res)
    elif exp_params["method"]=="Polyquery":
        filtered_params = filter_arguments(polyquery, new_params)
        eval_res = polyquery(dataset_df, **filtered_params)
        save_results(new_params, output_filename, **eval_res)
    # elif exp_params["method"]=="Polyquery_msed":
    #     filtered_params = filter_arguments(polyquery_msed, new_params)# Filter out unnecessary arguments
    #     eval_res = polyquery_msed(dataset_df, **filtered_params)    # Call  function with filtered arguments
    #     save_results(new_params, **eval_res)     
    elif exp_params["method"]=="Polyquery_msed_logscale":
        filtered_params = filter_arguments(polyquery_msed_logscale, new_params)
        eval_res = polyquery_msed_logscale(dataset_df, **filtered_params)
        save_results(new_params,output_filename, **eval_res)
    elif exp_params["method"]=="SVM":
        filtered_params = filter_arguments(svm, new_params)
        eval_res = svm(dataset_df, **filtered_params)
        save_results(new_params,output_filename, **eval_res)
    else:
        print(f"WARNING: Method {exp_params['method']} not implemented")


def ged_id_for_dict(dict):
    unique_string=''.join([f"{k}:{v};" for k,v in sorted(dict.items())]) 
    return hashlib.sha1(unique_string.encode()).hexdigest()

def write_exp_params(exp_params,f_todo):
      exp_id=ged_id_for_dict(exp_params)
      exp_params["exp_id"]=exp_id
      f_todo.write(json.dumps(exp_params)+"\n")