import pandas as pd
import re
import fasttext 
import numpy as np
from thefuzz import fuzz
import jaro
import networkx as nx
import matplotlib.pyplot as plt

from xgbClassifier import*


def preprocess(text: str) -> str:
    '''
    Preprocesses the given text by removing unwanted characters and formatting.

    Args:
        text (str): The input string to preprocess.

    Returns:
        str: The preprocessed string, cleaned and formatted.
    '''
    # This will remove other addition characters in the name of affiliation
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip().lower()

def read_data(path: str) -> pd.DataFrame:
    '''
    Read the data and pair the affiliation to find the matches
    Args:
        path(str): the path name of the file 

    Returns: the paired affiliation dataset

    '''
    df = pd.read_csv(path)
    pairing = []
    for i in range(len(df['name'])):
        for j in range(i + 1, len(df['name'])):
            # if i != j: this is not necessary because we did i+1 already makes sure it won't be equal
            pairing.append([df['name'][i], df['name'][j], i, j])
    dataframe = pd.DataFrame(pairing, columns=['affiliation1', 'affiliation2', 'rownumber1', 'rownumber2'])
    dataframe['affiliation1'] = dataframe['affiliation1'].apply(preprocess)
    dataframe['affiliation2'] = dataframe['affiliation2'].apply(preprocess)
    return dataframe

def generate_vector(sent: str) -> np.ndarray:
    '''
    Embeds each affiliation into a vector using a FastText model.

    Args:
        sent (str): The affiliation text.

    Returns:
        np.ndarray: The generated vector for the affiliation.
    '''
    path = "/Users/rura/Downloads/wiki.en/wiki.en.bin"
    ft = fasttext.load_model(path)
    
    # Generate and return the sentence vector
    return ft.get_sentence_vector(sent)

def euclideanDistance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    '''
    Calculates the Euclidean distance between two vectors that represent the affiliations.

    Args:
        vec1 (np.ndarray): The vector for the first affliation 
        vec2 (np.ndarray): The vector for the second affliation 

    Returns:
        float: The Euclidean distance between first affiliation  and second affiliation.
    '''
    return np.linalg.norm(vec1 - vec2)

def wRatio(string1: str, string2: str) -> int:
    '''
    Calculates the WRatio fuzzy matching score between two strings, 
    taking into account capitalization and punctuation differences.

    Args:
        string1 (str): The first affiliation.
        string2 (str): The second affiliation.

    Returns:
        int: A similarity score between 0 and 100, where 100 represents an exact match.
    '''
    return fuzz.WRatio(string1, string2)

def sortedtokenRatio(string1: str, string2: str) -> int:
    '''
    Calculates the fuzzy matching score between two strings, 
    accounting for word order but ignoring capitalization and punctuation.

    Args:
        string1 (str): The first affiliation.
        string2 (str): The second affiliation.

    Returns:
        int: A similarity score between 0 and 100, where 100 represents an exact match 
             after sorting words alphabetically.
    '''
    return fuzz.token_sort_ratio(string1, string2)

def settokenRatio(string1: str, string2: str) -> int:
    '''
    Calculates the fuzzy matching score between two strings, 
    accounting for word order and ignoring duplicates, capitalization, and punctuation.

    Args:
        string1 (str): The first affiliation.
        string2 (str):The second affiliation.

    Returns:
        int: A similarity score between 0 and 100, where 100 represents an exact match 
             after sorting words alphabetically and ignoring duplicates.
    '''
    return fuzz.token_set_ratio(string1, string2)

def jwDistance(string1: str, string2: str) -> float:
    '''
    Calculates the Jaro-Winkler distance between two strings, 
    measuring their similarity.

    Args:
        string1 (str): The first affiliation
        string2 (str): The second affiliation

    Returns:
        float: A similarity score between 0.0 and 1.0, where 1.0 represents an exact match.

    '''
    return jaro.jaro_winkler_metric(string1, string2)

def create_deduplication_map(matches: pd.DataFrame) -> pd.DataFrame:
    '''
    Finds connected components in the input DataFrame, assigns a unique matched_id 
    to each group, and generates a deduplication map for matched and duplicate IDs.

    Args:
        matches (pd.DataFrame): DataFrame with pairs representing matching connections.

    Returns:
        pd.DataFrame: A deduplication map with 'matched_id' and 'duplicate_id' columns.
    '''
    # Create a graph from the edgelist DataFrame
    G = nx.from_pandas_edgelist(matches, "rownumber1", "rownumber2")

    # Find connected components
    components = [list(component) for component in nx.connected_components(G)]

    # Prepare the rows for deduplication_map
    rows = []
    for component in components:
        if len(component) > 1:  
            matched_id = min(component)  # Select the smallest ID as the matched_id
            for duplicate_id in component:
                if matched_id != duplicate_id:
                    rows.append({
                        'matched_id': matched_id,
                        'duplicate_id': duplicate_id,
                    })

    # Create and filter the final deduplication_map DataFrame
    deduplication_map = pd.DataFrame(rows)

    return deduplication_map



if __name__ == "__main__":
    df = read_data("/Users/rura/Downloads/affiliation.csv")
    #print(df)
    df["aff1_vec"] = df["affiliation1"].apply(generate_vector)
    df["aff2_vec"] = df["affiliation2"].apply(generate_vector)

    # read in the training data for the classifier 
    small_df = pd.read_csv("/Users/rura/Downloads/aff_pairs.csv")
    small_df["aff1_vec"] = small_df["affiliation1"].apply(generate_vector)
    small_df["aff2_vec"] = small_df["affiliation2"].apply(generate_vector)
    small_df["eu_dist"] = small_df.apply(lambda x: euclideanDistance(x["aff1_vec"], x["aff2_vec"]), axis=1)
    
    # Convert columns to string
    #small_df["affiliation1"] = small_df["affiliation1"].astype(str)
    #small_df["affiliation2"] = small_df["affiliation2"].astype(str)
    
    # Apply the functions row-wise 
    small_df["wRatio"] = small_df.apply(lambda x: wRatio(x["affiliation1"], x["affiliation2"]), axis=1)
    small_df["sortRatio"] = small_df.apply(lambda x: sortedtokenRatio(x["affiliation1"], x["affiliation2"]), axis=1)
    small_df["setRatio"] = small_df.apply(lambda x: settokenRatio(x["affiliation1"], x["affiliation2"]), axis=1)
    small_df["jwDistance"] = small_df.apply(lambda x: jwDistance(x["affiliation1"], x["affiliation2"]), axis=1)
    print(small_df.head())


    train_x = small_df.drop(columns=["affiliation1", "affiliation2", "aff1_vec", "aff2_vec", "Matching "])
    train_y = small_df["Matching "]

    ## the classifier on small dataframe
    xgb = xgbClassier()
    xgb.fit(train_x, train_y)
    predict_train = xgb.predict(train_x)



    ### apply to big the dataframe df
    df["eu_dist"] = df.apply(lambda x: euclideanDistance(x["aff1_vec"], x["aff2_vec"]), axis=1)
    # Convert columns to string
    df["affiliation1"] = df["affiliation1"].astype(str)
    df["affiliation2"] = df["affiliation2"].astype(str)

    # Apply the functions row-wise 
    df["wRatio"] = df.apply(lambda x: wRatio(x["affiliation1"], x["affiliation2"]), axis=1)
    df["sortRatio"] = df.apply(lambda x: sortedtokenRatio(x["affiliation1"], x["affiliation2"]), axis=1)
    df["setRatio"] = df.apply(lambda x: settokenRatio(x["affiliation1"], x["affiliation2"]), axis=1)
    df["jwDistance"] = df.apply(lambda x: jwDistance(x["affiliation1"], x["affiliation2"]), axis=1)

    ## testing classifier
    test_x = df.drop(columns=["affiliation1", "affiliation2", "aff1_vec", "aff2_vec", "rownumber1", "rownumber2"])
    
    # predict the target on the test dataset
    predict_test = xgb.predict(test_x)

    ## create a new column in clean_df with the predicted matching
    df["predicted_match"] = predict_test

    ## filter the matching pairs
    matches = df[df["predicted_match"] == 1][["rownumber1", "rownumber2"]]
    matching_row = create_deduplication_map(matches)
    print(matching_row)

