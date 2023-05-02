import pandas as pd
import re
def preprocess(text):
    # this will remove other addition character in the name of affiliation
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip().lower()

def read_data(path: str) -> pd.DataFrame:
    '''
    Args:
        path(str): the path name

    Returns: the affiliation dataset

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

if __name__ == "__main__":
    df = read_data("/Users/rura/Downloads/affiliation.csv")
    print(df)

