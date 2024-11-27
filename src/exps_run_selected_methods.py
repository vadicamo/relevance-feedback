import os
import json
import pandas as pd
import numpy as np
from f_process_data import *
from f_run_exps import *
from tqdm import tqdm
import multiprocessing
import concurrent.futures
import time

"""
### Note on Experimental Parameters ###

This script contains all the tested cases as described in the paper entitled "Comparative Analysis of Relevance Feedback Techniques for Image Retrieval". 
Running all the experiments may take a considerable amount of time. 
We suggest that you first select only the method, number of seeds (for display initialization), 
and the type of feedback you are interested in for an initial test. 

By narrowing down your selections, you can expedite the experimentation process 
and focus on specific areas of interest.

Thank you for your understanding, and happy experimenting!
"""

### SELECT PARAMETERS #### 
#select the methods to be tested (this affects the execution time)
methods_to_test=["SVM", "Rocchio", "Polyquery", "Polyquery_msed_logscale"]

#select the update methods to be tested (this affects the execution time)
# true means that the batch update method is used
# false means that the incremental  update method is used
batch_values=[True, False]

# Seeds from 0 to 10 for reproducibility
seeds = list(range(10))

# List of n_display values to test
n_displays_list = [50]  # e.g., [50, 100]

# List of max_iter values to test (maximum number of iterations)
max_iter_list = [10]  # e.g., [10, 20]

# Pairs of k_pos and k_neg values to test
k_pos_neg_list = [(-1, 0), (3, 0), (5, 0), (-1, -1), (3, 3), (5, 5)]
#k_pos=-1, k_neg=0 (user at each iteration select all the diplayed relavant images and no non relevant images)
#k_pos=-1, k_neg=-1 (user at each iteration select all the diplayed relavant and non relevant images)
#k_pos=2, k_neg=0 (user at each iteration select 2 relavant images and no non relevant images)
#k_pos=2, k_neg=2 (user at each iteration select 2 relavant images and 2 non relevant images)
#k_pos=5, k_neg=0 (user at each iteration select 5 relavant images and no non relevant images)
#k_pos=5, k_neg=5 (user at each iteration select 5 relavant images and 5 non relevant images) etc

# List of datasets to test
datasets = [ "df_1608.csv", "df_1707.csv", "df_1597.csv", "df_1671.csv", "df_1598.csv", "df_1714.csv", "df_1711.csv", "df_1746.csv", "df_1725.csv", "df_1678.csv", "df_1739.csv", "df_1606.csv", "df_1733.csv", "df_1710.csv", "df_1676.csv", "df_1720.csv", "df_1736.csv", "df_1747.csv", "df_1744.csv", "df_1713.csv", "df_1593.csv", "df_1738.csv", "df_1750.csv", "df_1728.csv", "df_1672.csv", "df_1719.csv", "df_1702.csv", "df_1717.csv", "df_1748.csv", "df_1664.csv", "df_1665.csv", "df_1667.csv", "df_1669.csv", "df_1741.csv", "df_1609.csv", "df_1701.csv", "df_1729.csv", "df_1724.csv", "df_1607.csv", "df_1715.csv", "df_1675.csv", "df_1591.csv", "df_1668.csv", "df_1735.csv", "df_1670.csv", "df_1677.csv", "df_1673.csv", "df_1594.csv", "df_1595.csv", "df_1663.csv", "df_1604.csv", "df_1705.csv", "df_1680.csv", "df_1666.csv", "df_1610.csv", "df_1712.csv", "df_1732.csv", "df_1601.csv", "df_1731.csv", "df_1592.csv", "df_1703.csv", "df_1704.csv", "df_1706.csv", "df_1734.csv", "df_1721.csv", "df_1708.csv", "df_1661.csv", "df_1726.csv", "df_1602.csv", "df_1718.csv", "df_1709.csv", "df_1603.csv", "df_1740.csv", "df_1599.csv", "df_1723.csv", "df_1749.csv", "df_1737.csv", "df_1745.csv", "df_1600.csv", "df_1605.csv"]


### PATHS AND FILENAMES ###

# folder where the output files are saved
output_directory="out_results/"


# dictionary with the output filenames
output_filenames={
"SVM": f"{output_directory}/exps_res_svm.jsonl",
"Rocchio": f"{output_directory}/exps_res_rocchio.jsonl",
"Polyquery": f"{output_directory}/exps_res_poly.jsonl",
"Polyquery_msed_logscale": f"{output_directory}/exps_res_poly_msed_logscale.jsonl"
}

#dataset paths
normalized_zip_path = "data/dataset_normalized.zip"
logistic_zip_path = "data/dataset_logistic.zip"
softmax_zip_path = "data/dataset_softmax.zip"

# Starting position images for display 0
starting_pos_images_display0=np.load("data/Index_quintuplets.npy").tolist()

### PRECOMPUTED VALUES ###

#average of the distance between a data object and radom diplayed images (precomputed value)
avg_d_norm=(8.210553)

#average of the similarity  between a data object aand radom diplayed images (precomputed value)
avg_s_norm= (2.3878746)





def read_csv_from_zip(zip_path, internal_file):
    """
    Reads a CSV file from a given zip archive without extracting it.
    
    Args:
        zip_path (str): The path to the zip file.
        internal_file (str): The path to the CSV file inside the zip.

    Returns:
        pd.DataFrame: The dataframe loaded from the CSV file.
    """
   
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if internal_file in zip_ref.namelist():
            with zip_ref.open(internal_file) as file:
                return pd.read_csv(file)
        else:
            raise FileNotFoundError(f"'{internal_file}' not found in the zip archive.")
        

def setup_output_files(method):
    # Set up output file paths for a given method.
    output_filename = output_filenames[method]
    output_err_filename = output_filename.split(".")[0] + "_err.jsonl"
    output_done_filename = output_filename.split(".")[0] + "_done.jsonl"
    output_todo_fn = output_filename.split(".")[0] + "_todo.jsonl"



    # Create the output_done_filename if it does not exist.
    if not os.path.exists(output_done_filename):
        open(output_done_filename, 'w').close()

    return output_filename, output_err_filename, output_done_filename, output_todo_fn


def create_experiment_tasks(output_todo_fn, seeds, n_displays_list, k_pos_neg_list, datasets, max_iter_list, starting_pos_images_display0,avg_d_norm):
    # Create a file with all the parameters of the experiments.
    with open(output_todo_fn, 'w') as f_todo:
        for n_display in n_displays_list:
            for seed in seeds:
                for k_pos, k_neg in k_pos_neg_list:
                    for dataset in datasets:
                        dataset_name = dataset.split(".")[0]
                        for max_iter in max_iter_list:
                            indexes_positive_initial_images = starting_pos_images_display0[seed]
                            exp_params = {
                                "dataset": dataset_name,
                                "seed": seed,
                                "max_iter": max_iter,
                                "n_display": n_display,
                                "k_pos": k_pos,
                                "k_neg": k_neg,
                                "indexes_positive_initial_images": indexes_positive_initial_images,
                            }
                            process_methods(exp_params, f_todo,avg_d_norm)

def process_methods(exp_params, f_todo,avg_d_norm):
    # Process methods and write experimental parameters to the todo file.
    for method in methods_to_test:
        for use_batch in batch_values:
            if method == "Polyquery":
                process_polyquery(exp_params, use_batch, f_todo)
            elif method == "Polyquery_msed_logscale":
                process_polyquery_msed_logscale(exp_params, use_batch, f_todo)
            elif method == "Rocchio":
                process_rocchio(exp_params, use_batch, f_todo)
            elif method == "Pichunter":
                process_pichunter(exp_params, use_batch, f_todo,avg_d_norm)
            elif method == "SVM":
                process_svm(exp_params, use_batch, f_todo)
            else:
                print(f"Method {method} not implemented - skipping it")



    
def process_polyquery(exp_params, use_batch, f_todo):
    # Process Polyquery method-specific parameters.
    for data_preprocessing in ["softmax"]: #logistic_L1_normalized
        for fun_name in ["sed"]: #"triangular",
            new_exp_params = get_params(exp_params, "Polyquery", fun_name, data_preprocessing, use_batch)
            new_exp_params["alpha"], new_exp_params["beta"], new_exp_params["gamma"] = get_poly_alpha_beta_gamma(exp_params["k_neg"])
            write_exp_params(new_exp_params, f_todo)

def process_polyquery_msed_logscale(exp_params, use_batch, f_todo):
    # Process Polyquery_msed_logscale method-specific parameters.
    data_preprocessing = "softmax"
    new_exp_params = get_params(exp_params, "Polyquery_msed_logscale", "msed", data_preprocessing, use_batch)
    new_exp_params["alpha"], new_exp_params["beta"], new_exp_params["gamma"] = get_poly_msed_alpha_beta_gamma(exp_params["k_neg"])
    write_exp_params(new_exp_params, f_todo)

def process_rocchio(exp_params, use_batch, f_todo):
    # Process Rocchio method-specific parameters.
    data_preprocessing = "L2_normalized"
    for fun_name in ["euclidean"]:
        new_exp_params = get_params(exp_params, "Rocchio", fun_name, data_preprocessing, use_batch)
        new_exp_params["alpha"], new_exp_params["beta"], new_exp_params["gamma"] = get_rocchio_alpha_beta_gamma(exp_params["k_neg"], use_batch)
        write_exp_params(new_exp_params, f_todo)

def process_pichunter(exp_params, use_batch, f_todo):
    # Process Pichunter method-specific parameters.
    data_preprocessing = "L2_normalized"
    for m in [0.1, 1, 10, 100]:
        fun_name = "softmin"
        new_exp_params = get_params(exp_params, "Pichunter", fun_name, data_preprocessing, use_batch)
        new_exp_params["temperature"] = m * avg_d_norm
        new_exp_params["norm_multiplier"] = m
        write_exp_params(new_exp_params, f_todo)     

def process_svm(exp_params, use_batch, f_todo):
    # Process SVM method-specific parameters.
    data_preprocessing = "L2_normalized"
    new_exp_params = get_params(exp_params, "SVM", " ", data_preprocessing, use_batch)
    new_exp_params["gamma"] = 0
    new_exp_params["alpha"], new_exp_params["beta"] = get_svm_alpha_beta(exp_params["k_neg"], use_batch)
    write_exp_params(new_exp_params, f_todo)

def should_run_experiment(exp_params, done_experiments_df):
    # Check if the experiment should be run.
    id_exp = exp_params["exp_id"]
    return done_experiments_df.empty or (id_exp not in done_experiments_df["exp_id"].values)

def load_dataset(exp_params , data_preprocessing):
    # Load the dataset from the zip files.
    dataset_name = exp_params["dataset"]
    normalized_file = f"dataset_normalized/{dataset_name}.csv"
    logistic_file = f"dataset_logistic/{dataset_name}_logistic.csv"
    softmax_file = f"dataset_softmax/{dataset_name}_softmax.csv"

    # Read the CSV files directly from the zip files
    if data_preprocessing == "softmax":
        dataset_df= read_csv_from_zip(softmax_zip_path, softmax_file)
    elif data_preprocessing == "logistic_L1_normalized":
        dataset_df=  read_csv_from_zip(logistic_zip_path, logistic_file)
    elif data_preprocessing == "L2_normalized":
        dataset_df=  read_csv_from_zip(normalized_zip_path, normalized_file)
    else:
        print(f"Data preprocessing method {data_preprocessing} not implemented - skipping it")
        return None




    # Shuffle the columns of the normalized_df using seed 0 to avoid bias.
    img_ids = dataset_df.columns.tolist()
    shuffle_ids = np.random.RandomState(0).permutation(img_ids)
    dataset_df=  dataset_df[shuffle_ids]    
    return dataset_df



def run_experiments_for_method(method, output_filename, output_done_filename, output_todo_fn,output_err_filename):
    # Read done and todo experiments, then run the required ones.

    with open(output_done_filename, 'r') as f_done:
        done_experiments = f_done.read().splitlines() 
        done_experiments = [json.loads(exp) for exp in done_experiments]
        #transform the list of dictionaries in a  dataframe to make the search faster
        done_experiments_df=pd.DataFrame(done_experiments)
        f_done.close()

    with open(output_todo_fn, 'r') as f_todo:
            todo_experiments = f_todo.read().splitlines() 
            todo_experiments = [json.loads(exp) for exp in todo_experiments]
            f_todo.close()
    
   
    with open(output_done_filename, 'a') as f_done, open(output_err_filename, 'w') as f_err:
        print(f"Running experiments for {method}")
        for exp_params in tqdm(todo_experiments, desc=f"Processing  {method}", leave=True, miniters=2):
            if should_run_experiment(exp_params, done_experiments_df):
                dataset_df = load_dataset(exp_params,exp_params["data_preprocessing"])
                try:
                    run_experiment(exp_params, output_filename, dataset_df)
                    f_done.write(json.dumps(exp_params) + "\n")
                except Exception as e:
                    # Log the experiment parameters and the error message in the error file
                    error_entry = {
                        "error": str(e),
                        "experiment_params": exp_params
                    }
                    f_err.write(json.dumps(error_entry) + "\n")
        
        
        #delete the error file if it is empty
        if os.stat(output_err_filename).st_size == 0:
            os.remove(output_err_filename)
            os.remove(output_todo_fn)#delete the todo file
        else:
            print(f"Error file {output_err_filename} for {method} is not empty. Check it for more details.Adiing skipping experiments in the future todo file")
            #add the experiments that have failed to the todo file
            with open(output_err_filename, 'r') as f_err:
                error_experiments = f_err.read().splitlines() 
                error_experiments = [json.loads(exp) for exp in error_experiments]
                f_err.close()
            with open(output_todo_fn, 'w') as f_todo:
                for exp_params in error_experiments:
                    write_exp_params(exp_params, f_todo)
                f_todo.close()
        print(f"Experiments for method {method} COMPLETED")
        pass
            

def run_method_experiment(method):
    output_filename, output_err_filename, output_done_filename, output_todo_fn = setup_output_files(method)
    create_experiment_tasks(output_todo_fn, seeds, n_displays_list, k_pos_neg_list, datasets, max_iter_list, starting_pos_images_display0, avg_d_norm)
    run_experiments_for_method(method, output_filename, output_done_filename, output_todo_fn, output_err_filename)
    pass


def main():
    print(os.getcwd())
    print("Starting experiments for methods:", methods_to_test)
    print ("Update methods to test (True=Batch, False= Incremental):", batch_values)
    print("Output files:", [output_filenames[m] for m in methods_to_test])

    print("Seeds:", seeds)
    print("n_displays_list:", n_displays_list)
    print("max_iter_list:", max_iter_list)
    print("k_pos_neg_list:", k_pos_neg_list)
    print("avg_d_norm:", avg_d_norm)
    print("avg_s_norm:", avg_s_norm)
    print("Number of datasets:", len(datasets))
    # Create the output directory if it does not exist.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
 
    # #Option 1: Run the experiments sequentially
    #  #iterate over the methods to be tested 
    # for method in methods_to_test:
    #     run_method_experiment(method)


    #Option 2: Run the experiments in parallel
    # Determine the number of processes to use: the minimum of the number of methods and available CPUs.
    num_processes = min(len(methods_to_test), multiprocessing.cpu_count())

    # Create a pool with the determined number of processes.
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use `tqdm` to track progress, applying `imap` for efficient mapping of processes.
        results = list(tqdm(pool.imap(run_method_experiment, methods_to_test), 
                            total=len(methods_to_test), 
                            desc="Methods"))
   
    print("Experiments completed (check err files if they exists in the output folder).")

if __name__ == '__main__':
      
      main()