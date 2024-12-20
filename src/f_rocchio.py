
import numpy as np
import pandas as pd
from f_similarity_metrics import get_distance_matrix, get_similarity_matrix
from f_evaluation_metrics import calculate_ap_for_iterations, calculate_ndcg_for_iterations, calculate_recall_at_k_for_iterations
from f_display_and_feedback import get_actions, sample_first_display,create_display
from f_process_data import get_gt_dict




def new_query_rocchio(query_old, relevant, non_relevant, alpha, beta, gamma):
    '''
    Parameters
    ----------
    query_old: old query
    relevant: relevant images
    non_relevant: non relevant images
    alpha: alpha parameter
    beta: beta parameter
    gamma: gamma parameter

    '''
    if relevant.shape[1] == 0:
        centroid_relevant = np.zeros((query_old.shape[0], 1))
    else:
        if len(relevant.shape) == 1:
            centroid_relevant = relevant.reshape(-1, 1)
        else:
            centroid_relevant = np.mean(relevant, axis=1).reshape(-1, 1)
    if non_relevant.shape[1] == 0:
        centroid_non_relevant = np.zeros((query_old.shape[0], 1))
    else:
        if len(non_relevant.shape) == 1:
            centroid_non_relevant = non_relevant.reshape(-1, 1)
        else:
            centroid_non_relevant = np.mean(non_relevant, axis=1).reshape(-1, 1)

    new_query = alpha * query_old + beta * centroid_relevant - gamma * centroid_non_relevant

    return  new_query



def rocchio_single_step(data_df,display_df, relevant_ids,non_relevant_ids, alpha=1, beta=1, gamma=1,fun_name="euclidean", initial_query=None):
    '''
    Parameters
    ----------
    data_df : DataFrame
        DataFrame with the dataset one column for each image 
    display_df : DataFrame with the old display
    relevant_ids : list of indexes of the relevant images 
    non_relevant_ids : list of indexes of the non relevant images
    alpha : float, optional alpha parameter
    beta : float, optional beta parameter
    gamma : float, optional gamma parameter
    fun_name : string, optional function to calculate the similarity or metric model
    initial_query : initial query to start the Rocchio algorithm
    
    '''
    #n_display is the number of columns in the display_df
    n_display=display_df.shape[1]
       
    old_query=initial_query

    # Initialize the query with zero if not provided
    if initial_query is None:
        old_query = np.zeros((data_df.shape[0], 1))
        # beta=beta+alpha #we want to keep alpha + beta - gamma=1
        # alpha=0


    selected_images_at_this_iteration=[im for im in non_relevant_ids]+[im for im in relevant_ids]
    
    if len(selected_images_at_this_iteration)==0:
        return display_df, initial_query
        
    relevant = data_df[relevant_ids].to_numpy()
    non_relevant = data_df[non_relevant_ids].to_numpy() 
    new_query = new_query_rocchio(old_query, relevant, non_relevant, alpha, beta, gamma)
    old_query = new_query


    if(fun_name=="euclidean") or (fun_name=="triangular") or (fun_name=="jsd") or (fun_name=="sed"):
        distance_vector= get_distance_matrix(pd.DataFrame(new_query), data_df, fun_name=fun_name)
        display_df = create_display(data_df, distance_vector, n_display, is_ascending=True)
    elif(fun_name=="dotproduct") or (fun_name=="cosine"):
        similarity_vector= get_similarity_matrix(pd.DataFrame(new_query), data_df, fun_name=fun_name)
        display_df = create_display(data_df, similarity_vector, n_display, is_ascending=False)
    else:
        print(f"ERR The function {fun_name} is not implemented yet. Using Euclidean distance.")  
        distance_vector= get_distance_matrix(pd.DataFrame(new_query), data_df, fun_name="euclidean")
        display_df = create_display(data_df, distance_vector, n_display, is_ascending=True)

    return display_df, new_query
       


def rocchio(df, indexes_positive_initial_images, seed=0,  n_display=25, max_iter=20, alpha=1, beta=1, gamma=1,fun_name="euclidean", k_pos=-1, k_neg=-1,  initial_query=None, use_batch=False):
    '''
    Parameters
    ----------
    df :
      DataFrame with the dataset one column for each image and the last row with the GT relevance.
    indexes_positive_initial_images : 
        list of indexes of the relevant images in the first display. 
    seed : 
        int, optional seed for the random sampling
    n_display : 
        int, optional number of images in the display
    max_iter: maximum number of iterations
    plot_images_bool: boolean to plot the images
    fun_name: function to calculate the similarity or metric model
    k_pos: number of positive images to select
    k_neg: number of negative images to select
    iterations_for_evaluation: list of iterations to evaluate the model
    initial_query: initial query to start the Rocchio algorithm
    use_batch: boolean to use the batch version of the Rocchio algorithm
    '''
    

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

    #using rocchio to compute the new query and the new display
    display_df, new_query= rocchio_single_step(data_df,display_df, relevant_ids,non_relevant_ids, alpha, beta, gamma,fun_name=fun_name, initial_query=initial_query)
    old_query = new_query

    #iter 1
    iter_count += 1   

    
    while num_positive_in_the_display < n_display and iter_count < max_iter and action != []:
        action, list_display_relevance = get_actions(display_df, groundtruth_dic,already_selected_images, k_pos, k_neg,no_reclick_image=use_batch)
        already_selected_images += [im for im, _ in action]
        action_dic[iter_count]=action
        num_positives_at_iter.append(sum(list_display_relevance)) 
        iteration_display_relevance.append(list_display_relevance)
        num_positive_in_the_display=sum(list_display_relevance)

        
        #using rocchio to computre the new query and the new display
        if use_batch:
            relevant_ids+=[im for im, rel in action if rel == 1]
            non_relevant_ids+=[im for im, rel in action if rel == 0]
            display_df, new_query= rocchio_single_step(data_df,display_df, relevant_ids,non_relevant_ids, alpha, beta, gamma,fun_name=fun_name, initial_query=initial_query)
        else:
            relevant_ids=[im for im, rel in action if rel == 1]
            non_relevant_ids=[im for im, rel in action if rel == 0]
            display_df, new_query= rocchio_single_step(data_df,display_df, relevant_ids,non_relevant_ids, alpha, beta, gamma,fun_name=fun_name, initial_query=old_query)
            
        old_query = new_query

        iter_count += 1 #increment the iteration counter and create the new display
      
    while iter_count <= max_iter:
        iteration_display_relevance.append(list_display_relevance)
        num_positives_at_iter.append(sum(list_display_relevance)) 
        action_dic[iter_count]=[]
        iter_count += 1

    map_results=calculate_ap_for_iterations(iteration_display_relevance)
    ndcg_results = calculate_ndcg_for_iterations(iteration_display_relevance )
    recall_results = calculate_recall_at_k_for_iterations(iteration_display_relevance,k=n_display)
    
    res={"map_results":map_results, "ndcg_results":ndcg_results, "recall_results":recall_results, "num_positives_at_iter":num_positives_at_iter, "action_dic":action_dic}
    return res



