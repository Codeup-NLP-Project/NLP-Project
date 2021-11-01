
import unicodedata
import re, os
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

# ### 1. Define a function named basic_clean. It should take in a string and apply some basic text cleaning to it:
# Lowercase everything
# Normalize unicode characters
# Replace anything that is not a letter, number, whitespace or a single quote.

def basic_clean(string:str) -> str:
    '''
    This function takes in a string and
    returns the string normalized.
    '''
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    string = re.sub(r'[^\w\s]', '', string).lower()
    return string


# ### 2. Define a function named `tokenize`. It should take in a string and tokenize all the words in the string.
def tokenize(string:str) -> str:
    '''
    This function takes in a string and
    returns a tokenized string.
    '''
    # Create tokenizer.
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    # Use tokenizer
    string = tokenizer.tokenize(string, return_str = True)
    
    return string


# ### 3. Define a function named `stem`. It should accept some text and return the text after applying stemming to all the words.

def stem(string:str) -> str:
    '''
    This function takes in a string and
    returns a string with words stemmed.
    '''
    # Create porter stemmer.
    ps = nltk.porter.PorterStemmer()
    
    # Use the stemmer to stem each word in the list of words we created by using split.
    stems = [ps.stem(word) for word in string.split()]
    
    # Join our lists of words into a string again and assign to a variable.
    string = ' '.join(stems)
    
    return string


# ### 4. Define a function named `lemmatize`. It should accept some text and return the text after applying lemmatization to each word.

def lemmatize(string:str) -> str:
    '''
    This function takes in string for and
    returns a string with words lemmatized.
    '''
    # Create the lemmatizer.
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Use the lemmatizer on each word in the list of words we created by using split.
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    # Join our list of words into a string again and assign to a variable.
    string = ' '.join(lemmas)
    
    return string


# ### 5. Define a function named `remove_stopwords`. It should accept some text and return the text after removing all the stopwords.
# 
# ### This function should define two optional parameters, extra_words and exclude_words. These parameters should define any additional stop words to include, and any words that we don't want to remove.

def remove_stopwords(string:str, extra_words = [], exclude_words = []) -> str:
    '''
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    '''
    # Create stopword_list.
    stopword_list = stopwords.words('english')
    
    # Remove 'exclude_words' from stopword_list to keep these in my text.
    stopword_list = set(stopword_list) - set(exclude_words)
    
    # Add in 'extra_words' to stopword_list.
    stopword_list = stopword_list.union(set(extra_words))

    # Split words in string.
    words = string.split()
    
    # Create a list of words from my string with stopwords removed and assign to variable.
    filtered_words = [word for word in words if word not in stopword_list]
    
    # Join words in the list back into strings and assign to a variable.
    return ' '.join(filtered_words)


# ### 8. For each dataframe, add the following columns:
# > * `original` to hold the orginial text
# > * `stemmed` to hold the stemmed version of the cleaned data.
# > * `lemmatized` to hold the lemmatized version of the cleaned data.
# > * The column data will contain the cleaned data

def prep_readme_data(df:pd.DataFrame, column:str, extra_words=[], exclude_words=[]) -> pd.DataFrame:
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and eturns a df
    with the columns clean, stemmed, and lemmatized text with stopwords removed.
    '''
    # Set the originial to the originial text
    df['original'] = df[column]

    # Clean the dataframe within the input column
    df['cleaned'] = df[column].apply(basic_clean).apply(tokenize).apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['stemmed'] = df[column].apply(basic_clean).apply(tokenize).apply(stem).apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['lemmatized'] = df[column].apply(basic_clean).apply(tokenize).apply(lemmatize).apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    return df

def find_dominant_lang(s: str, threshold:float) -> str:
    '''Takes a string of programming languages and associated 
    percentages and will return the one that is above a certain threshold
    percentage so that is most representative of that programming language.
    
    Must ensure the threshold > 50
    '''
    # find the languages from the raw string
    l = re.findall(r"([A-Z].*?)'", s)
    # find the percentages from the raw string
    p = re.findall(r"(\d*\.\d*)", s)
    
    # Iterate though the percentages to check if it's above threshold
    for n in range(len(p)):
        # Try to convert to float and check
        try:
            # Check if percentage composition gte to threshold
            if float(p[n]) >= threshold:
                # Return the language
                return l[n]
        # If there is an error, just return None
        except:
            return None
        
    # If no language was greater than the threshold, return None
    return None

def get_readme_data(threshold = 75, extra_words=[], exclude_words=[]) -> pd.DataFrame:
    '''Takes threshold for when a programming language is representative of that README
    text. The threshold need to be > 50. Additionally adds the cleaned columns of
    'orginial', 'stemmed', 'lemmatized' from the 'readme' column where the readme
    column is now the CLEANED text. This also checks if the cashed file exists or not
    if it doesn't it will be created.

    Additional options that can be passed are the extra_words and exclude_words
    which would be passed to the cleaning function and perform their associated 
    definitions.
    '''
    # names the file with the defined threshold at the end with the decimal
    # replaced if it is present
    filename = f"clean_readme_{str(threshold).replace('.', '_')}.csv"

    if os.path.exists(filename):
        return pd.read_csv(filename)
    
    # Let's you know if the file wasn't found
    print(f'Did not find the file {filename}')

    # Pull in the source readme data
    df = pd.read_csv('readme_data_c.csv')
    # Filter out all readmes that do not meet the threshold
    df['prog_lang'] = df.programming_language.apply(lambda x: find_dominant_lang(x, threshold))
    # Set the dataframe to the languages that returned a value and 
    # reset the index and remove other columns
    df = (df[~df.prog_lang.isna()].reset_index().drop(columns=[
        'programming_language', 'parsed', 'user_repo', 'index']))
    
    # Let's the user know that the data is being cleaned and to be patient
    print('cleaning data, hold your horses....')

    # Sends the dataframe and with column readme as the column to be cleaned with the
    # parameters of extra_words and exclude_words as optional parameters.
    df = prep_readme_data(df, 'readme', extra_words=extra_words, exclude_words=exclude_words)

    # Save the dataframe to the filename
    df.to_csv(filename)
    # Let's user know that the file was successfully saved and the name
    print(f'saved file: {filename}')

    # Return the dataframe that has been filtered and cleaned
    return df


