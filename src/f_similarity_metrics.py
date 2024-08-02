from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import additive_chi2_kernel
from scipy.stats import entropy
import numpy as np



def softmax(x,temperature=1, axis=1):
    '''
    Parameters:
    x: input array 
    temperature: temperature parameter
    Returns:
    softmax: softmax of the input data along each axis-slice of x
    '''    
    e_x = np.exp(x/temperature)
    return e_x / e_x.sum(axis=axis, keepdims=True)

def logistic(x,temperature=1, axis=1):
    '''
    Parameters:
    x: input array 
    temperature: temperature parameter
    Returns:
    num: logistic of the input array along each axis-slice of x
    '''
    num=1/(1+np.exp(-x/temperature))
    return num/num.sum(axis=axis, keepdims=True)


# def complexity(x,axis=1):
#     '''
#     Parameters:
#     x: input probabilistic array 
#     Returns:
#     complexity: complexity of the input array
#     '''
#     #entropy is the shannon entropy
#     return np.exp(entropy(x, axis=axis))


def sed(x, y):
    '''
    Parameters:
    x: input probabilistic array 
    y: input probabilistic array 
    Returns:
    sed: symmetric entropy distance along each axis-slice of x  the two input arrays
    '''
    return np.exp(jensenshannon(x,y))-1

def jensenshannon(x, y):
    '''
    Parameters:
    x: input probabilistic array
    y: input probabilistic array
    Returns:
    jensenshannon: Jensen-Shannon divergence along each axis-slice of  the two input arrays

    '''
   # Compute the average of the two arrays
    avg = (x + y) / 2
    # Compute the shannon_entropy of the average array
    shannonentropy_of_avg = entropy(avg)
    #compute the average of the shannon_entropy of the two arrays
    avg_of_shannonentropy = (entropy(x) + entropy(y)) / 2.0
    
    return  (shannonentropy_of_avg-avg_of_shannonentropy)


# def jensen_shannon_matrix_test(x_df, y_df):
#     if isinstance(x_df, np.ndarray):
#         x_matrix = x_df
#     else:
#         x_matrix = x_df.values

#     if isinstance(y_df, np.ndarray):
#         y_matrix = y_df
#     else:
#         y_matrix = y_df.values

#     results = np.zeros((x_matrix.shape[0], y_matrix.shape[0]))
    
#     for i in range(x_matrix.shape[0]):
#         for j in range(y_matrix.shape[0]):
#             results[i, j] = jensenshannon(x_matrix[i], y_matrix[j])
            
#     return results

def jensen_shannon_matrix(x_df, y_df):
    if isinstance(x_df, np.ndarray):
        x_matrix = x_df
    else:
        x_matrix = x_df.values

    if isinstance(y_df, np.ndarray):
        y_matrix = y_df
    else:
        y_matrix = y_df.values

    results = np.zeros((x_matrix.shape[0], y_matrix.shape[0]))
    
    for i in range(x_matrix.shape[0]):
        for j in range(y_matrix.shape[0]):
            results[i, j] = jensenshannon(x_matrix[i], y_matrix[j])
            
    return results

def sed_matrix(x_df, y_df):
    if isinstance(x_df, np.ndarray):
        x_matrix = x_df
    else:
        x_matrix = x_df.values

    if isinstance(y_df, np.ndarray):
        y_matrix = y_df
    else:
        y_matrix = y_df.values

    results = np.zeros((x_matrix.shape[0], y_matrix.shape[0]))
    
    for i in range(x_matrix.shape[0]):
        for j in range(y_matrix.shape[0]):
            results[i, j] = sed(x_matrix[i], y_matrix[j])
            
    return results


def get_similarity_matrix(data1_df, dataset_df,fun_name="dotproduct"): 
    '''
    data1_df: DataFrame with the queries or display
    dataset_df: DataFrame with the dataset
    similarity_fun_name: similatity function to calculate 

    '''
    # Function to get the similarity matrix between the display and the whole dataset
    n = data1_df.shape[1]
    m = dataset_df.shape[1]
    similarity = np.zeros((m, n))
    if  fun_name == "shifted_dotproduct":
        similarity=1+ np.dot(dataset_df.T, data1_df) #.values
    elif fun_name == "dotproduct":
        similarity= np.dot(dataset_df.T, data1_df)
    elif fun_name == "shifted_cosine":
        similarity = 1 + cosine_similarity(dataset_df.T, data1_df.T)
    elif fun_name == "cosine":
        similarity = cosine_similarity(dataset_df.T, data1_df.T)
    elif fun_name == "jsd":
        similarity= 1-jensen_shannon_matrix(dataset_df.T, data1_df.T) 
    elif fun_name == "triangular":
        similarity= 1+0.5*additive_chi2_kernel(dataset_df.T, data1_df.T) # 1-triangular_matrix_matrix(dataset_df.T, data1_df.T)
    elif fun_name== "sed":
        similarity= 1-sed_matrix(dataset_df.T, data1_df.T)
    else: 
         print(f"ERR The function {fun_name} is not implemented. Using shifted cosine similarity.")  
         similarity = 1+ cosine_similarity(dataset_df.T, data1_df.T)
 
    return similarity


def get_similarity_matrix_temperature(data1_df, dataset_df,user_model_fun_name="softmin", temperature=1): #only for picHunter
    '''
    data1_df: DataFrame with the display
    dataset_df: DataFrame with the dataset
    user_model_fun_name: function to calculate the user model

    '''
    # Function to get the similarity matrix between the display and the whole dataset
    n = data1_df.shape[1]
    m = dataset_df.shape[1]
    similarity = np.zeros((m, n))
    #switch case to select the function to calculate the user model

    if user_model_fun_name == "softmin":
        similarity = np.exp(-euclidean_distances(dataset_df.T, data1_df.T)/temperature)
    elif user_model_fun_name == "softmax_cosine":
        similarity = np.exp(cosine_similarity(dataset_df.T, data1_df.T)/temperature)
    elif user_model_fun_name == "l1_normalized_cosine":
        similarity =(1.001+cosine_similarity(dataset_df.T, data1_df.T))/temperature  
    else: 
        print(f"ERR The user model function {user_model_fun_name} is not implemented. Using softmin similarity.")  
        similarity = np.exp(-euclidean_distances(dataset_df.T, data1_df.T)/temperature)
 
    return similarity 

# def get_soft_similarity_matrix(data1_df, dataset_df,user_model_fun_name="softmin", temperature=1): 
#     '''
#     data1_df: DataFrame with the display
#     dataset_df: DataFrame with the dataset
#     user_model_fun_name: function to calculate the user model

#     '''
#     similarity=get_similarity_matrix_temperature(data1_df, dataset_df,user_model_fun_name=user_model_fun_name, temperature=temperature)
 
#     return similarity/similarity.sum(axis=1)[:, np.newaxis] #row-wise normalized version of the similarity scores between the columns of dataset_df and data1_df


def get_distance_matrix(data1_df, dataset_df, fun_name="euclidean"): 
    '''
    data1_df: DataFrame with the data queries or all display
    dataset_df: DataFrame with the dataset
    fun_name: similatity function to calculate 

    '''
    # Function to get the similarity matrix between the display and the whole dataset
    n = data1_df.shape[1]
    m = dataset_df.shape[1]
    distance_matrix = np.zeros((m, n))
    #switch case to select the function to calculate the user model
    if  fun_name == "euclidean":
        distance_matrix=euclidean_distances(dataset_df.T, data1_df.T)
    elif fun_name == "triangular":
        distance_matrix= -0.5*additive_chi2_kernel(dataset_df.T, data1_df.T) 
    elif fun_name == "jsd":
        distance_matrix= jensen_shannon_matrix(dataset_df.T, data1_df.T)
    elif fun_name == "sed":
        distance_matrix= sed_matrix(dataset_df.T, data1_df.T)
    else: 
         print(f"ERR The function {fun_name} is not implemented. Using Euclidean distance.")  
         distance_matrix=euclidean_distances(dataset_df.T, data1_df.T)
 
    return distance_matrix

