import numpy as np

def calculate_average_precision(relevance_scores):
    '''
    Calculate the  Average Precision for for a list of relevance scores
   
     Parameters:
        relevance_scores (list): A list of relevance scores for retrieved items.
            Each score should be either 0 (irrelevant) or 1 (relevant).

     Returns:
        float: The  Average Precision value.

    '''
    num_relevant_retrieved = 0 # Number of relevant items encountered so far
    precision_at_k = [] # Sum of Precision@K values
    #num_relevant = len(relevance_scores)  # Total number of retrieved relevant items

    for k, relevance in enumerate(relevance_scores, start=1):
        if relevance == 1:
            num_relevant_retrieved += 1
        precision_at_k.append(num_relevant_retrieved / k)

    if num_relevant_retrieved == 0:
        return 0.0 
    else:
        return np.mean(precision_at_k)
    

def calculate_ndcg(relevance_scores):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG)
    Parameters:
        relevance_scores (list): A list of relevance scores for retrieved items.
            Each score should be either 0 (irrelevant) or 1 (relevant).

     Returns:
        float: The NDCG value.
    """
    #ideal_ranking is a list of 1 same lenght as relevance_scores
    ideal_ranking = [1] * len(relevance_scores) #here we shoudl cosidere the whole dataset not only the display, so I assumed taht in teh dataset we have a number of relevant > nDisplay
    #ideal_ranking = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg(ideal_ranking)
    if ideal_dcg == 0:
        return 0.0
    
    return dcg(relevance_scores) / ideal_dcg

def dcg(relevance_scores):
    """
    Calculate the Discounted Cumulative Gain (DCG)
    Parameters:
        relevance_scores (list): A list of relevance scores for retrieved items.
            Each score should be either 0 (irrelevant) or 1 (relevant).

     Returns:
        float: The DCG value.
    """
    # dcg = 0.0
    # for i, relevance  in enumerate(relevance_scores, start=1):
    #     dcg += (2**relevance - 1) / np.log2(i + 1)    
    return np.sum((2**np.array(relevance_scores) - 1) / np.log2(np.arange(1, len(relevance_scores) + 1) + 1))

def calculate_recall_at_k(relevance_scores, k):
    """
    Calculate the recall for a given iteration.
    Parameters:
        relevance_scores (list): A list of relevance scores for retrieved items.
            Each score should be either 0 (irrelevant) or 1 (relevant).
    Returns:
        float: The recall value.
    """
    relevant = sum(relevance_scores)  # Total number of relevant retrieved items
    retrieved = len(relevance_scores)   # Total number of retrieved items
    k=min(k,retrieved)
    return relevant / k if k > 0 else float('nan')

def calculate_ap_for_iterations(iteration_display_relevance, iterations=[]):
    """
    Calculate the  Average Precision (AP) for the selected iterations. 
    Parameters:
        iteration_display_relevance (list): A list of relevance scores for each iteration.
        iterations (list): A list of iteration indices for which to calculate the MAP. 
            If not provided, AP will be calculated for all iterations.
    Returns:    
        list: A list of AP values for the selected iterations.
    """
    if not iterations:
        iterations = range(len(iteration_display_relevance)+1)

    ap_results = []
    
    for index in iterations:
        if index < len(iteration_display_relevance):
            map_value = calculate_average_precision(iteration_display_relevance[index])
        else:
            map_value = float('nan')
        ap_results.append(map_value)
    
    return ap_results

def calculate_ndcg_for_iterations(iteration_display_relevance,iterations=[]):
    """
    Calculate NDCG for the selected iterations
    Parameters:
        iteration_display_relevance (list): A list of relevance scores for each iteration.
        iterations (list): A list of iteration indices for which to calculate the NDCG. 
            If not provided, NDCG will be calculated for all iterations.
    Returns:
        list: A list of NDCG values for the selected iterations.
    """
    if not iterations:
        iterations = range(len(iteration_display_relevance)+1)

    ndcg_results = []

    for index in iterations:
        if index < len(iteration_display_relevance):
            ndcg_value = calculate_ndcg(iteration_display_relevance[index])
        else:
            ndcg_value = float('nan')
        ndcg_results.append(ndcg_value)
    
    return ndcg_results



def calculate_recall_at_k_for_iterations(iteration_display_relevance, k, iterations=[]):
    """
    Calculate the recall for the selected iterations
    Parameters:
        iteration_display_relevance (list): A list of relevance scores for each iteration.
        iterations (list): A list of iteration indices for which to calculate the recall. 
            If not provided, recall will be calculated for all iterations.
    Returns:
        list: A list of recall values for the selected iterations.

    """
   
    if not iterations:
        iterations = range(len(iteration_display_relevance)+1)
        
    recall_results = []
    
    for index in iterations:
        if index < len(iteration_display_relevance):
            recall_value = calculate_recall_at_k(iteration_display_relevance[index],k)
        else:
            recall_value = float('nan')
        recall_results.append(recall_value)
    
    return recall_results
