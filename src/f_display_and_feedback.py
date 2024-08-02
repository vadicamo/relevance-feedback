import pandas as pd
import numpy as np

def get_actions(display_df_t, groundtruth_dic, already_selected_images, k_pos=-1, k_neg=-1, no_reclick_image=True):
    '''
    Parameters:
    display_df_t: DataFrame with the display one column for each image  
    groundtruth_dic: dictionary with the groundtruth relevance of the images
    already_selected_images: list of images already selected (cannot be selected again))
    k_pos: number of positive images to select
    k_neg: number of negative images to select
    no_reclick_image: boolean to avoid reclicking on the same image
    
    Returns:
    actions: list of tuples with the selected images and their relevance
    list_display_relevance: list of the relevance of the images in the display
    '''
    # Function to get the action vector from a display
    positive_action = []
    negatives_action = []
    list_display_relevance=[] #store the relevance of the images in the display
    for image in display_df_t.columns:
        list_display_relevance.append(groundtruth_dic[image])
        if no_reclick_image:
            if (groundtruth_dic[image] == 1) and (image not in already_selected_images):
                positive_action.append(image)
            if (groundtruth_dic[image] == 0) and (image not in already_selected_images):
                negatives_action.append(image)
        else:
            if (groundtruth_dic[image] == 1): #not used in teh experiments but might be useful to test in the incremental case
                positive_action.append(image)
            if (groundtruth_dic[image] == 0):
                negatives_action.append(image)



    #if k_pos !=-1 and k_pos is smaller than the number of positive images in the display we randomly select k_pos positive images 
    if k_pos !=-1  and k_pos < len(positive_action):
        np.random.shuffle(positive_action) #random shaffle the positive images and select the first k_pos
        positive_action = positive_action[:k_pos]

    #if k_neg !=-1 and k_neg is smaller than the number of negative images in the display we randomly select k_neg negative images
    if k_neg != -1 and k_neg < len(negatives_action):
        np.random.shuffle(negatives_action) #random shaffle the negative images and select the first k_neg
        negatives_action = negatives_action[:k_neg]
    
    positive_actions=[(action, 1) for action in positive_action]
    negative_actions=[(action, 0) for action in negatives_action]
    actions = positive_actions + negative_actions
    return actions, list_display_relevance 

def sample_first_display(df, seed,n_display, indexes_positive_initial_images):
    '''
    Parameters:
    df: DataFrame with the dataset one column for each image and the last row with the GT relevance
    seed: int,  seed for the random sampling
    n_display: int,  number of images in the display
    indexes_positive_initial_images: list of indexes of the relevant images in the first display.
    
    Returns:
    first_display: DataFrame with the first display
    '''
    # Function to sample and transpose the display dataset
    #np.random.seed(seed)  # select the seed for numpy
    selected_columns = df.columns[df.iloc[-1, :] == 1] 
    #filtered_df = df.loc[:, selected_columns]
    #positive_display_df=filtered_df[filtered_df.iloc[:, indexes_positive_initial_images]]  # select  relevant images to be shown in the display (using indexes_positive_initial_pairs)
    selected_columns=selected_columns[indexes_positive_initial_images]
    positive_display_df = df.loc[:, selected_columns]
    n_positive=len(indexes_positive_initial_images)
    n_negative=max(n_display-n_positive,0)

    non_relevant_df = df.loc[:, df.iloc[-1] == 0] #non_relevant images   
    negative_display_df = non_relevant_df.sample(n=n_negative, axis=1,  random_state=seed) # select n_display-2 non_relevant images to be shown in the display
    first_display = pd.concat([positive_display_df, negative_display_df], axis=1).iloc[:-1, :]
    return first_display

def create_display(data_df, score, n_display, is_ascending=False):
    '''
    Parameters:
    data_df: DataFrame with the dataset one column for each image and the last row with the GT relevance
    score: list with the score of each image
    n_display: int, number of images in the display
    is_ascending: boolean, to sort the display in ascending order
    
    Returns:
    display_df: DataFrame with the display
    '''
    # Function to create and transpose the display dataset
    df_copy_t = data_df.copy().transpose()   
    df_copy_t['score'] = score
    df_sorted = df_copy_t.sort_values(by='score', ascending=is_ascending)
    display_df_t = df_sorted.head(n_display)
    display_df_t = display_df_t.copy()
    display_df_t.drop(columns=['score'], errors='ignore', inplace=True)
    display_df = display_df_t.transpose()
    return display_df
