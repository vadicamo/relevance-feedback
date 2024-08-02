import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from f_process_data import get_gt_dict
from f_display_and_feedback import get_actions, sample_first_display,create_display
from f_evaluation_metrics import calculate_ap_for_iterations, calculate_ndcg_for_iterations, calculate_recall_at_k_for_iterations

def get_msed_logscale_sim(data_df,entropy_dict,relevant_ids,non_relevant_ids,old_score,alpha,beta,gamma,dict):
    ''' 
    Parameters:
    data_df: DataFrame with the dataset one column for each image
    entropy_dict: dictionary with the entropy of each image, image_id are the key
    relevant_ids: list of relevant images
    non_relevant_ids: list of non relevant images
    beta: beta parameter
    gamma: gamma parameter
        dict : dictionary with the precomputed values
        precomputed_sum_pos: precomputed sum of the positive images + data
        precomputed_sum_neg: precomputed sum of the negative images +data
        precomputed_sum_entropy_neg: precomputed sum of the entropy of the negative images and the data complexity
        precomputed_sum_entropy_pos: precomputed sum of ethentropy  of the positive images and the data complexity
    Returns:
    new_values: list of new values
    '''     

    msed_score=[] 

    precomputed_sum_neg=dict['precomputed_sum_neg']
    precomputed_sum_pos=dict['precomputed_sum_pos']
    precomputed_sum_entropy_neg=dict['precomputed_sum_entropy_neg']
    precomputed_sum_entropy_pos=dict['precomputed_sum_entropy_pos']
    n_pos=dict['n_pos']
    n_neg=dict['n_neg']

    for el in relevant_ids : 
        precomputed_sum_pos+=data_df[el].to_numpy().reshape(1, -1)
        precomputed_sum_entropy_pos+=entropy_dict[el]
        n_pos+=1


    for el in non_relevant_ids: 
        precomputed_sum_neg+=data_df[el].to_numpy().reshape(1, -1)
        precomputed_sum_entropy_neg+=entropy_dict[el]
        n_neg+=1
        
    new_dict={'precomputed_sum_pos':precomputed_sum_pos,
              'precomputed_sum_neg':precomputed_sum_neg,
              'precomputed_sum_entropy_neg':precomputed_sum_entropy_neg,
              'precomputed_sum_entropy_pos':precomputed_sum_entropy_pos,
              'n_pos':n_pos,
              'n_neg':n_neg}
            
    #iterate over data_df
     
    for img_id, row in data_df.T.iterrows():
        data_vec=row.to_numpy().reshape(1, -1)
        data_entropy=entropy_dict[img_id]  
        
        mean_pos=(precomputed_sum_pos+ data_vec)/(n_pos+1)
        entropy_numerator_pos=shannon_entropy(mean_pos,axis=1)[0]
        avg_entropy_denominator_pos=(precomputed_sum_entropy_pos+data_entropy)/(n_pos+1)
        score_pos= n_pos+1-np.exp(entropy_numerator_pos-avg_entropy_denominator_pos)

        mean_neg=(precomputed_sum_neg+ data_vec)/(n_neg+1)
        entropy_numerator_neg=shannon_entropy(mean_neg,axis=1)[0]
        avg_entropy_denominator_neg=(precomputed_sum_entropy_neg+data_entropy)/(n_neg+1)
        score_neg= n_neg+1-np.exp(entropy_numerator_neg-avg_entropy_denominator_neg)


        score= beta*score_pos-gamma*score_neg
        msed_score.append(score)
        

    new_scores=alpha*old_score+np.array(msed_score).flatten()

    return new_scores, new_dict
    

def get_msed_logscale_sim_vec(data_df,entropy_dict,query):
    '''
    Parameters:
    data_df: DataFrame with the dataset one column for each image
    entropy_dict: dictionary with the entropy of each image, image_id are the key
    query: query vector
    Returns:
    msed_values: list of new values

    '''
    # m = df_with_complexity.shape[0]
    # msed_values = np.zeros((m, 1))       

    msed_score=[] 

    query_complexity= np.exp(shannon_entropy(query,axis=1))# complexity 

    for index, row in data_df.iterrows():
        data_vec=row.to_numpy().reshape(1, -1)
        sum_pos= query+data_vec
        mean_pos=sum_pos/(2.0)
        complexity_pos_num=np.exp(shannon_entropy(mean_pos, axis=1))
        complexity_pos_product_den=query_complexity*np.exp((entropy_dict[index]))
        score_pos= 2- complexity_pos_num/ np.sqrt(complexity_pos_product_den)
   
        msed_score.append(score_pos)
        print(f"score_pos: {score_pos} type: {type(score_pos)}")

    return np.array(msed_score).flatten()

def polyquery_msed_logscale(df, indexes_positive_initial_images, seed=0,  n_display=25, max_iter=20, alpha=1, beta=0.7, gamma=0.7, k_pos=-1, k_neg=-1,  initial_query=None, use_batch=False):
    '''
    Parameters
    ----------
    df : DataFrame
        DataFrame with the dataset one column for each image and the last row with the GT relevance.
    indexes_positive_initial_images : list of indexes of the relevant images in the first display. 
    seed : int, optional seed for the random sampling
    n_display : int, optional number of images in the display
    max_iter: maximum number of iterations
    plot_images_bool: boolean to plot the images
    fun_name: function to calculate the similarity or metric model
    k_pos: number of positive images to select
    k_neg: number of negative images to select

    initial_query: ID of initial query to start the algortitm 
    use_batch: boolean to use the batch version of the  algorithm
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


                                 
    display_df, new_score,precomputed_dict,entropy_dict= poly_msed_logscale_single_step(data_df,display_df, relevant_ids,non_relevant_ids,precomputed_dict_initial=None, alpha=alpha, beta=beta, gamma=gamma, initial_query=initial_query,entropy_dict=None)
    old_score = new_score

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
            display_df, new_score,precomputed_dict,entropy_dict= poly_msed_logscale_single_step(data_df,display_df, relevant_ids,non_relevant_ids,precomputed_dict_initial=None, alpha=alpha, beta=beta, gamma=gamma, initial_query=initial_query,entropy_dict=entropy_dict)
        else:
            relevant_ids=[im for im, rel in action if rel == 1]
            non_relevant_ids=[im for im, rel in action if rel == 0]
            display_df, new_score,precomputed_dict,entropy_dict= poly_msed_logscale_single_step(data_df,display_df, relevant_ids,non_relevant_ids,precomputed_dict_initial=precomputed_dict, alpha=alpha, beta=beta, gamma=gamma, initial_scores=old_score,entropy_dict=entropy_dict)

        old_score = new_score
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
 
def poly_msed_logscale_single_step(data_df,display_df, relevant_ids,non_relevant_ids,precomputed_dict_initial=None, alpha=0.7, beta=0.7, gamma=0.4, initial_query=None, initial_scores=None,entropy_dict=None):
    '''
    Parameters
    ----------
    data_df : DataFrame
        DataFrame with the dataset one column for each image
    display_df : DataFrame
        DataFrame with the old display one column for each image
    relevant_ids : list of relevant images
    non_relevant_ids : list of non relevant images
    precomputed_dict_initial : dictionary with the precomputed values
    alpha : float, optional
        DESCRIPTION. The default is 0.7.
    beta : float, optional
        DESCRIPTION. The default is 0.7.
    gamma : float, optional
        DESCRIPTION. The default is 0.4.
    initial_query : TYPE, optional
        DESCRIPTION. The default is None.
    initial_scores : TYPE, optional 
        DESCRIPTION. The default is None.
    entropy_dict : dictionary, optional
        dictionary with the entropy of each image, image_id are the key
    Returns
    -------
    display_df : DataFrame
        new diplay
    new_scores : list
        new scores for each image
    precomputed_dict : dictionary
        updated dictionary with the precomputed values
    entropy_dict : dictionary
        dictionary with the entropy of each image

    '''
    #n_display is the number of columns in the display_df
    n_display=display_df.shape[1]


    if entropy_dict is None:
       #create a dataframe with the complexity of each image
        entropy_dict={}
        for index, row in data_df.T.iterrows():
            #data_complexity_df.loc[index,'data']= row.to_numpy().reshape(1, -1)
            entropy_dict[index]=shannon_entropy(row)
        

    if precomputed_dict_initial is None:
        precomputed_dict={"precomputed_sum_pos":0, "precomputed_sum_neg":0, 
                        "precomputed_sum_entropy_neg":1,
                        "precomputed_sum_entropy_pos":1,
                        "n_pos":0, 
                        "n_neg":0}
    else:
        precomputed_dict=precomputed_dict_initial.copy()
    
    old_scores=initial_scores
    # Initialize the query with zero if not provided
    if old_scores is None:
        if initial_query is None:
            old_scores= np.array([0] *data_df.shape[1] ) 
            # beta=beta+alpha #we want to keep alpha + beta - gamma=1
            # alpha=0
        else:
            old_scores = get_msed_logscale_sim_vec(data_df,entropy_dict,initial_query)

    selected_images_at_this_iteration=[im for im in non_relevant_ids]+[im for im in relevant_ids]

    if len(selected_images_at_this_iteration)==0:
        return display_df, old_scores,precomputed_dict,entropy_dict
   
    
              
    new_scores, precomputed_dict= get_msed_logscale_sim(data_df,entropy_dict,relevant_ids,non_relevant_ids,old_scores,alpha,beta,gamma,precomputed_dict)
    display_df = create_display(data_df, new_scores, n_display, is_ascending=False)

    return display_df, new_scores, precomputed_dict, entropy_dict
