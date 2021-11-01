
import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

# ### 1. Define a function named basic_clean. It should take in a string and apply some basic text cleaning to it:
# 
# > * Lowercase everything
# > * Normalize unicode characters
# > * Replace anything that is not a letter, number, whitespace or a single quote.
# 



def basic_clean(string):
    '''
    This function takes in a string and
    returns the string normalized.
    '''
    string = unicodedata.normalize('NFKD', string)             .encode('ascii', 'ignore')             .decode('utf-8', 'ignore')
    string = re.sub(r'[^\w\s]', '', string).lower()
    return string


# ### 2. Define a function named `tokenize`. It should take in a string and tokenize all the words in the string.

# In[40]:


def tokenize(string):
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

# In[46]:


def stem(string):
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

# In[50]:


def lemmatize(string):
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

# In[20]:


def remove_stopwords(string, extra_words = [], exclude_words = []):
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


# ### 8. For each dataframe, produce the following columns:
# 
# > * `title` to hold the title
# > * `original` to hold the original article/post content
# > * `clean` to hold the normalized and tokenized original with the stopwords removed.
# > * `stemmed` to hold the stemmed version of the cleaned data.
# > * `lemmatized` to hold the lemmatized version of the cleaned data.

# In[58]:


def prep_article_data(df, column, extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    df['clean'] = df[column].apply(basic_clean)                            .apply(tokenize)                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['stemmed'] = df[column].apply(basic_clean)                            .apply(tokenize)                            .apply(stem)                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    df['lemmatized'] = df[column].apply(basic_clean)                            .apply(tokenize)                            .apply(lemmatize)                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
    
    return df