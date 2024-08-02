import numpy as np
import pandas as pd
from f_similarity_metrics import get_similarity_matrix
from f_evaluation_metrics import calculate_ap_for_iterations, calculate_ndcg_for_iterations, calculate_recall_at_k_for_iterations
from f_display_and_feedback import get_actions, sample_first_display,create_display
from f_process_data import get_gt_dict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def distance_from_hyperplane(w, b, data_df):
    return np.dot(data_df.T, w) + b

def svm_score(data_df, relevant, non_relevant):
    '''
    Parameters
    ----------
    relevant: relevant images
    non_relevant: non relevant images
    '''

    # X  Training vectors, i.e. teh union of relevant and non_relevant, y=Target values (class labels in classification='1' for relevant  or '0' for non-relevant).
    X = np.concatenate((relevant, non_relevant), axis=0)
    y = np.concatenate((np.ones(relevant.shape[0]), np.zeros(non_relevant.shape[0])))
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    accuracy = accuracy_score(y, clf.predict(X))
    distances_to_hyper = distance_from_hyperplane(w, b, data_df)
    
    return  distances_to_hyper

def svm_single_step(data_df,display_df, relevant_ids,non_relevant_ids,initial_scores=None,alpha=0, beta=1):
    n_display=display_df.shape[1]

    
    old_scores=initial_scores
    # Initialize the query with zero if not provided
    if old_scores is None:
        old_scores= (np.array([0] * data_df.shape[1]) )

    if len(relevant_ids)==0: 
         return display_df,old_scores

    
    relevant = data_df[relevant_ids].to_numpy().T
    non_relevant = data_df[non_relevant_ids].to_numpy().T
   
    new_scores = alpha*old_scores+beta*svm_score(data_df, relevant, non_relevant)
    display_df = create_display(data_df, new_scores, n_display, is_ascending=False)
    
    return display_df, new_scores

def svm(df, indexes_positive_initial_images, seed=0,  n_display=25, max_iter=20,  k_pos=-1, k_neg=-1,  use_batch=False,alpha=0, beta=1):
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
    k_pos: number of positive images to select
    k_neg: number of negative images to select, if 0 teh alogoritmh select 10 random image not in the display as negative
    iterations_for_evaluation: list of iterations to evaluate the model
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
    action, list_display_relevance = get_actions(display_df, groundtruth_dic, already_selected_images, k_pos, k_neg)
    already_selected_images= [im for im, _ in action]
    relevant_ids=[im for im, rel in action if rel == 1]
    non_relevant_ids=[im for im, rel in action if rel == 0]
    fake_non_relevant_ids=[]
    if len(non_relevant_ids)==0 : ##SVM need to have at least one negative image in the display
        #select nfake random images df that are not in df_display 
        fake_non_relevant_ids = [im for im in data_df.columns if (im not in display_df.columns) and (im not in already_selected_images)]
        np.random.shuffle(fake_non_relevant_ids)
        fake_non_relevant_ids = fake_non_relevant_ids[:10]
        negative_actions=[(action, 0) for action in fake_non_relevant_ids]
        action = action + negative_actions
        action_dic[iter_count]=action
        
        
    action_dic[iter_count]=action
    num_positives_at_iter.append(sum(list_display_relevance)) 
    iteration_display_relevance.append(list_display_relevance)
    num_positive_in_the_display=sum(list_display_relevance)


    display_df,new_scores= svm_single_step(data_df,display_df, relevant_ids,non_relevant_ids+fake_non_relevant_ids,initial_scores=None,alpha=alpha, beta=beta)
    old_scores = new_scores

    #iter 1
    iter_count += 1  
    
    while num_positive_in_the_display < n_display and iter_count < max_iter and action != []:
        action, list_display_relevance = get_actions(display_df, groundtruth_dic,already_selected_images, k_pos, k_neg)
        already_selected_images += [im for im, _ in action]
        current_non_relevant_ids=[im for im, rel in action if rel == 0]
        fake_non_relevant_ids=[]
        if len(current_non_relevant_ids)==0: ##SVM need to have at least one negative image in the display
            #select 10 random images that are not in df_display 
            fake_non_relevant_ids = [im for im in data_df.columns if (im not in display_df.columns) and (im not in already_selected_images)]
            np.random.shuffle(fake_non_relevant_ids)
            fake_non_relevant_ids = fake_non_relevant_ids[:10]
            negative_actions=[(action, 0) for action in fake_non_relevant_ids]
            action = action + negative_actions
        

        action_dic[iter_count]=action
        num_positives_at_iter.append(sum(list_display_relevance)) 
        iteration_display_relevance.append(list_display_relevance)
        num_positive_in_the_display=sum(list_display_relevance)

        if use_batch:
            relevant_ids+=[im for im, rel in action if rel == 1]
            non_relevant_ids+=[im for im, rel in action if rel == 0]
            display_df,_= svm_single_step(data_df,display_df, relevant_ids,non_relevant_ids+fake_non_relevant_ids,initial_scores=None,alpha=alpha, beta=beta)
            
        else:
            relevant_ids=[im for im, rel in action if rel == 1]
            non_relevant_ids=[im for im, rel in action if rel == 0]        
            display_df,new_scores= svm_single_step(data_df,display_df, relevant_ids,non_relevant_ids+fake_non_relevant_ids, initial_scores=old_scores,alpha=alpha, beta=beta)
     
        old_scores = new_scores
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
      


