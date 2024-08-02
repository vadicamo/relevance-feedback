
import os
import json
import pandas as pd
import numpy as np
from f_process_data import *
from f_run_exps import *
from tqdm import tqdm


##############################################
#### Eperimental parameters #######
#seeds from 0 to 10 to make results reproducible, and select the same starting display for each method
seeds = list(range(10))
# a list of n_display values to test, where n_display is the number of images to display at each iteration 
n_displays_list = [50] #, 100
#  max_iter values to test, where max_iter is the maximum number of iterations to run the algorithm
max_iter_list = [10] #, 20
#pairs of k_pos and k_neg values to test, where k_pos is the number of positive images the user select at each iteration,
# and k_neg is the number of negative images the user select at each iteration
# -1 means that all the relevant /Non_relevant images are  are selected+
# 0 means that no relevant/Non_relevant images are selected
#case to test: 
#k_pos=-1, k_neg=0 (user at each iteration select all the diplayed relavant images and no non relevant images)
#k_pos=-1, k_neg=-1 (user at each iteration select all the diplayed relavant and non relevant images)
#k_pos=2, k_neg=0 (user at each iteration select 2 relavant images and no non relevant images)
#k_pos=2, k_neg=2 (user at each iteration select 2 relavant images and 2 non relevant images)
#k_pos=5, k_neg=0 (user at each iteration select 5 relavant images and no non relevant images)
#k_pos=5, k_neg=5 (user at each iteration select 5 relavant images and 5 non relevant images) etc
k_pos_neg_list =   [(-1,0),(3,0), (5,0), (-1,-1),(3,3),(5,5)]
avg_d_norm=(8.210553)
avg_s_norm= (2.3878746)


##############################################
#### Data path and info #######
datasets = [ "df_1608.csv", "df_1707.csv", "df_1597.csv", "df_1671.csv", "df_1598.csv", "df_1714.csv", "df_1711.csv", "df_1746.csv", "df_1725.csv", "df_1678.csv", "df_1739.csv", "df_1606.csv", "df_1733.csv", "df_1710.csv", "df_1676.csv", "df_1720.csv", "df_1736.csv", "df_1747.csv", "df_1744.csv", "df_1713.csv", "df_1593.csv", "df_1738.csv", "df_1750.csv", "df_1728.csv", "df_1672.csv", "df_1719.csv", "df_1702.csv", "df_1717.csv", "df_1748.csv", "df_1664.csv", "df_1665.csv", "df_1667.csv", "df_1669.csv", "df_1741.csv", "df_1609.csv", "df_1701.csv", "df_1729.csv", "df_1724.csv", "df_1607.csv", "df_1715.csv", "df_1675.csv", "df_1591.csv", "df_1668.csv", "df_1735.csv", "df_1670.csv", "df_1677.csv", "df_1673.csv", "df_1594.csv", "df_1595.csv", "df_1663.csv", "df_1604.csv", "df_1705.csv", "df_1680.csv", "df_1666.csv", "df_1610.csv", "df_1712.csv", "df_1732.csv", "df_1601.csv", "df_1731.csv", "df_1592.csv", "df_1703.csv", "df_1704.csv", "df_1706.csv", "df_1734.csv", "df_1721.csv", "df_1708.csv", "df_1661.csv", "df_1726.csv", "df_1602.csv", "df_1718.csv", "df_1709.csv", "df_1603.csv", "df_1740.csv", "df_1599.csv", "df_1723.csv", "df_1749.csv", "df_1737.csv", "df_1745.csv", "df_1600.csv", "df_1605.csv"]
#starting_pos_images_display0_fn= {-1:"Index_couples.npy", 2:"Index_couples.npy", 3:"Index_triplets.npy", 5:"Index_quintuplets.npy"}
#starting_pos_images_display0_dict = {i : np.load(starting_pos_images_display0_fn[i]) for i in starting_pos_images_display0_fn.keys()}
starting_pos_images_display0=np.load("Index_quintuplets.npy").tolist()
batch_values=[True, False]
output_filename=f"out_results/clip/exps_res_poly.jsonl"
output_err_filename=output_filename.split(".")[0]+"_err.jsonl"  
output_done_filename=output_filename.split(".")[0]+"_done.jsonl"  
output_todo_fn=output_filename.split(".")[0]+"_todo.jsonl"  
output_directory = os.path.dirname(output_filename)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
if not os.path.exists(output_done_filename):
    with open(output_done_filename, 'w') as f_done:
        f_done.close()
##############################################



if __name__ == '__main__':
    print(os.getcwd())
    print("Starting experiments")

    #creatiting a file with all the parameters of the experiments
   # if not os.path.exists(output_todo_fn):
    with open(output_todo_fn, 'w') as f_todo:
        for n_display in n_displays_list:
            for seed in seeds:
                for k_pos, k_neg in k_pos_neg_list:
                    for dataset in datasets:
                        dataset_name=dataset.split(".")[0]
                        for max_iter in max_iter_list:   
                            indexes_positive_initial_images= starting_pos_images_display0[seed]
                            exp_params={
                                        "dataset":dataset_name, 
                                        "seed":seed,
                                        "max_iter":max_iter, 
                                        "n_display":n_display, 
                                        "k_pos":k_pos, 
                                        "k_neg":k_neg,
                                        "indexes_positive_initial_images": indexes_positive_initial_images,
                                        }
                            for use_batch in batch_values:
                                method="Polyquery"
                                for data_preprocessing in ["softmax"]: #logistic_L1_normalized
                                    fun_to_test=["triangular", "sed"]
                                    for fun_name in fun_to_test:
                                        new_exp_params=get_params(exp_params,method,fun_name, data_preprocessing, use_batch)
                                        if k_neg==0:
                                            alpha=0.75
                                            beta=0.25
                                            gamma=0
                                        elif k_neg==-1:
                                            alpha=1
                                            beta=0.25
                                            gamma=0.25
                                        else:
                                            alpha=0.75
                                            beta=1
                                            gamma=0.75
                                        new_exp_params["alpha"]=alpha
                                        new_exp_params["beta"]=beta 
                                        new_exp_params["gamma"]=gamma 
                                        write_exp_params(new_exp_params,f_todo)
                                        
                               
                                                  
                                
                               
                            

    #iterating over the lines of  output_todo_fn, checking if the experiment has been already done or not in the output_done_filename
    #if not done, run the experiment and save the results in the output_filename
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

    with open(output_done_filename, 'a') as f_done:
       for exp_params in tqdm(todo_experiments):
                #check if the experiment has been already done
                id_exp=exp_params["exp_id"]
                 #check if done_experiments_df is not empty dataframe
                if (not done_experiments_df.empty) and (id_exp in done_experiments_df["exp_id"].values):
                    continue
                dataset_name=exp_params["dataset"]
                normalized_df = pd.read_csv("data/dataset_normalized/"+dataset_name+".csv")
                logistic_df = pd.read_csv("data/dataset_logistic/"+dataset_name+"_logistic.csv")
                softmax_df = pd.read_csv("data/dataset_softmax/"+dataset_name+"_softmax.csv")
                
                #shuffle the columns of the normalized_df using  seed 0 
                img_ids=normalized_df.columns.tolist()
                shuffle_ids=np.random.RandomState(0).permutation(img_ids)
                normalized_df=normalized_df[shuffle_ids]    
                logistic_df=logistic_df[shuffle_ids]
                softmax_df=softmax_df[shuffle_ids]
                dataset_df={"L2_normalized":normalized_df, "logistic_L1_normalized":logistic_df, "softmax":softmax_df}
                run_experiment(exp_params, output_filename, dataset_df[exp_params["data_preprocessing"]])
                f_done.write(json.dumps(exp_params)+"\n")

            

    

