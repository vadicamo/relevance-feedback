from readline import redisplay
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import numpy as np

from f_similarity_metrics import  get_similarity_matrix_temperature
from f_evaluation_metrics import calculate_ap_for_iterations, calculate_ndcg_for_iterations, calculate_recall_at_k_for_iterations
from f_display_and_feedback import get_actions, sample_first_display,create_display
from f_process_data import get_gt_dict



def pichunter_single_step(data_df,display_df, relevant_ids,non_relevant_ids,fun_name="softmin", initial_prob=0, temperature=1):
    '''
    Parameters
    ----------
    data_df: DataFrame with the dataset, one column for each image
    display_df: DataFrame with the display, one column for each image
    relevant_ids: list of relevant images
    non_relevant_ids: list of non relevant images
    fun_name: function to calculate the user model
    initial_prob: initial probability of each image being relevant
    temperature: temperature parameter for the softmax/min function
    
    Returns
    -------
    display_df: DataFrame with the new display
    new_prob_values: new probability values of each image being relevant
    
    
    '''
    n_display=display_df.shape[1]

    if (not isinstance(initial_prob, np.ndarray)) or (initial_prob is None):
        initial_prob =  np.full(data_df.shape[1],0.5) # prob of each image being relevant is 0.5
    

    # Calculate similarity and sissimilarity matrix
    similarity_matrix =get_similarity_matrix_temperature(display_df, data_df, user_model_fun_name=fun_name, temperature=temperature)
    dissimilarity_matrix=1/similarity_matrix #np.where(similarity_matrix!=0, 1/similarity_matrix, 1)
         
    #compute the soft similarity and dissimilarity matrix
    soft_similarity_matrix=similarity_matrix/similarity_matrix.sum(axis=1)[:, np.newaxis] 
    soft_dissimilarity_matrix=dissimilarity_matrix/dissimilarity_matrix.sum(axis=1)[:, np.newaxis]
    
    #create the dataframes with the soft similarity and dissimilarity matrices
    similarity_df = pd.DataFrame(soft_similarity_matrix, columns=display_df.columns, index=data_df.columns) #similarities to display images
    dissimilarity_df = pd.DataFrame(soft_dissimilarity_matrix, columns=display_df.columns, index=data_df.columns) #dissimilarities to display images
        
    # Calculate user model
    image_in_display=display_df.columns
    relevant_ids_in_display=[im for im in relevant_ids if im in image_in_display]
    non_relevant_ids_in_display=[im for im in non_relevant_ids if im in image_in_display]


    p_relevant=initial_prob #p(oi \in T, H_{t-1})
 

    prod_similartities_to_positive_actions= np.array(similarity_df[relevant_ids_in_display].product(axis=1))#similarity of each image to images selected as positive
    prod_dissimilarities_to_negative_actions=1
    if len(non_relevant_ids_in_display)>0:
        prod_dissimilarities_to_negative_actions= np.array(dissimilarity_df[non_relevant_ids_in_display].product(axis=1)) #dissimilarity of each image to images selected as negative

    usermodel_given_relevant= prod_similartities_to_positive_actions*prod_dissimilarities_to_negative_actions# (product over postive actions of dimilarities  * product over negative actions of dissim)
   
    numerator=usermodel_given_relevant*p_relevant
    denominator=numerator.sum()


    new_prob_values = numerator / denominator 
    display_df = create_display(data_df, new_prob_values, n_display)

    return display_df, new_prob_values

def pichunter(df,indexes_positive_initial_images,seed=0, n_display=25, max_iter=20, fun_name="softmin", k_pos=-1, k_neg=0, temperature=1, initial_prob=None, use_batch=False): 
    """
    Parameters
    ----------
    df: DataFrame with the dataset one column for each image and the last row with the GT relevance
    indexes_positive_initial_images : list of indexes of the relevant images in the first display. 
    seed : int, optional seed for the random sampling
    n_display : int, optional number of images in the display
    max_iter: maximum number of iterations
    plot_images_bool: boolean to plot the images
    fun_name: function to calculate the user model
    k_pos: number of positive images to select
    k_neg: number of negative images to select
    temperature: temperature parameter for the softmax/min function
    iterations_for_evaluation: list of iterations to calculate the evaluation metrics
    
    Returns
    -------
    res: dictionary with the evaluation metrics and the actions at each iteration
    
    """

 
    # Initialize variables
    iteration_display_relevance = [] #store the relevance of the images in the display at each iteration
    groundtruth_dic = get_gt_dict(df) # dictionary with the relevance of each image in the dataset
    #groutruth_vec = df.iloc[-1, :]
    data_df = df.iloc[:-1, :] #the dataset without the last row with the GT relevance
    num_positives_at_iter  = [] #store the number of positive images in the display at each iteration (used to compute recall)
    iter_count = 0 #initialize the iteration counter
    action_dic={} #store the actions at each iteration (first action is at iteration 0 since it is selecting images from teh first display)
    list_display_relevance=[]
    already_selected_images=[]

    #setting display at itertaion 0: we start from some positive images selected in the display and the rest are unlabeled 
    #display 0
    display_df = sample_first_display(df, seed, n_display, indexes_positive_initial_images)
    action, list_display_relevance = get_actions(display_df, groundtruth_dic, already_selected_images, k_pos, k_neg,no_reclick_image=use_batch)
    already_selected_images= [im for im, _ in action]
    relevant_ids=[im for im, rel in action if rel == 1]
    non_relevant_ids=[im for im, rel in action if rel == 0]
    action_dic[iter_count]=action
    num_positives_at_iter.append(sum(list_display_relevance)) 
    iteration_display_relevance.append(list_display_relevance)
    num_positive_in_the_display=sum(list_display_relevance)

    #using pichunter to computre the new query and the new display
    old_prob = initial_prob
    display_df, new_prob= pichunter_single_step(data_df,display_df, relevant_ids,non_relevant_ids,fun_name=fun_name, initial_prob=old_prob, temperature=temperature)
    old_prob = new_prob

    iter_count += 1
    
 
    
    while num_positive_in_the_display < n_display and iter_count < max_iter and action != []:
        action, list_display_relevance = get_actions(display_df, groundtruth_dic,already_selected_images, k_pos, k_neg,no_reclick_image=use_batch)
        already_selected_images += [im for im, _ in action]
        action_dic[iter_count]=action
        num_positives_at_iter.append(sum(list_display_relevance)) 
        iteration_display_relevance.append(list_display_relevance)
        num_positive_in_the_display=sum(list_display_relevance)
        if use_batch:
            relevant_ids+=[im for im, rel in action if rel == 1]
            non_relevant_ids+=[im for im, rel in action if rel == 0]
            display_df, new_prob= pichunter_single_step(data_df,display_df, relevant_ids,non_relevant_ids,fun_name=fun_name, initial_prob=old_prob, temperature=temperature) 
            old_prob = new_prob
        else:
            relevant_ids=[im for im, rel in action if rel == 1]
            non_relevant_ids=[im for im, rel in action if rel == 0]
            display_df, new_prob= pichunter_single_step(data_df,display_df, relevant_ids,non_relevant_ids,fun_name=fun_name, initial_prob=old_prob, temperature=temperature)
            old_prob = new_prob
        iter_count += 1
        

    while iter_count <= max_iter:
        iteration_display_relevance.append(list_display_relevance)
        num_positives_at_iter.append(sum(list_display_relevance)) 
        action_dic[iter_count]=[]
        iter_count += 1

       
    map_results=calculate_ap_for_iterations(iteration_display_relevance )
    ndcg_results = calculate_ndcg_for_iterations(iteration_display_relevance )
    recall_results = calculate_recall_at_k_for_iterations(iteration_display_relevance,k=n_display)
    res={"map_results":map_results, "ndcg_results":ndcg_results, "recall_results":recall_results, "num_positives_at_iter":num_positives_at_iter, "action_dic":action_dic}
    return res

