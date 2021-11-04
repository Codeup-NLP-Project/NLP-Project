#!/usr/bin/env python
# coding: utf-8

import re, os
import unicodedata
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import nltk.sentiment

from requests import get
from bs4 import BeautifulSoup

from wordcloud import WordCloud


plt.rc('figure', figsize=(13, 7))
plt.style.use('seaborn-darkgrid')


# ### This `fetch_user_followers_and_repos` function is to query a user's followers and the first 30 repositiories that they have publicly available on Github

def fetch_user_followers_and_repos(user: str):
    '''Takes a Github username as a string and will parse their github for followers
    and the users first upto 30 publicly available repos and returns 2 lists the followers and repositories
    '''
    # Set headers
    headers= {'User-Agent': 'Codeup Data Science'}
    # Set the parse_followers flag to True
    parse_followers = True
    # Start with page 1
    page = 1
    # Build a list to hold all the parsed followers
    all_users_followers = list()
    
    # Run through all pages of followers until there are not any more followers
    while parse_followers:
        # Runs to pull all followers from the user
        # Builds follower url to find the users follower's 
        follower_url = f'https://github.com/{user}?page={page}&tab=followers'
        print(follower_url)
        # Fetch response from url
        response = get(follower_url, headers=headers)

        # Return the user's followers html page
        follower_html = str(BeautifulSoup(response.text, 'html.parser'))

        # This regex pulls to pull followers usernames out of the html page
        # set as a set so it doesn't repeat users and remove the first char spot for the '/'
        user_followers = set([r[1][1:] for r in re.findall(r'(link_type\:self"\shref=")(.*?)"', follower_html)])
        
        # Add the new list of followers to the all_users_followers
        all_users_followers.extend(user_followers)
        
        # Check to see if there are not any more followers
        if not user_followers:
            parse_followers = False
        # Move onto the next page   
        page += 1
    
    # Build users repos
    repo_url = f'https://github.com/{user}?tab=repositories'
    
    # get the response from github repos
    response = get(repo_url, headers=headers) 
    
    # Return the user's repo html page
    repo_str = str(BeautifulSoup(response.text, 'html.parser'))
    
    # Filter the first page of repos and put into a set
    repos = set([r[1] for r in re.findall(
        r'(itemprop="name\scodeRepository">\n)\s*(.*?)<', repo_str)])
    
    return all_users_followers, repos


# ## Next we need to send the usernames to the function to pull all the information out

def add_followers_to_dataframe(df, followers: list):
    '''Takes a dataframe and a list of followers and checks if the followers are in the dataframe,
    if they are not, they will be added and set their default parsed value to False, returns
    amended dataframe
    '''
    # pull the followers from the dataframe and set to a list to search later
    followers_in_dataframe = df.index.to_list()
    
    # Iterate through the followers
    for follower in followers:
        # Check if the follower is in the dataframe
        if follower not in followers_in_dataframe:
            # Set the default parsed value to False
            df.loc[follower] = {'parsed': False}
    # Return the amended dataframe
    return df

def add_repos_to_readme_dataframe(user:str, repos:list):
    '''Takes the current user and the repos to search through and see if they are in the readme_dataframe index.
    If the repo is not in the readme_dataframe, it will add it to the index to pull the readme out later.
    '''
    # Set the filename for the readme
    filename = 'readme_data_c.csv'
    
    # Change the repos to a list so it can be scripted
    repos = list(repos)
    
    # Define the default columns for the dataframe
    default_cols ={
            'parsed': False, # Default parsed is False
            'readme': 'None', # Default readme is 'None'
            'programming_language': 'None' # Default programming language is 'None'
        }
    
    # Checks to see if the readme exists
    if os.path.exists(filename):
        # pull the readme_df in and set the index col to the user_repo
        readme_df = pd.read_csv(filename, index_col='user_repo')
    # If the readme_df does not exist, create it and set the default features
    else:
        # Add the user_repo to the default cols to set as index
        default_cols['user_repo'] = user + '/' + repos[0]
        # Define the dataframe
        readme_df = pd.DataFrame([default_cols]).set_index('user_repo')
        # Remove from default_cols so no key errors
        default_cols.pop('user_repo')
    
    # Iterate though the repos and check if they are in the readme_dataframe or not
    for repo in list(repos):
        # Combine user with repo to define the target index
        ind = user + '/' + repo
        # If ind is not in the readme file add it to it with the default cols
        if ind not in readme_df.index.to_list():
            # Add the user_repo to the readme with default_cols
            readme_df.loc[ind] = default_cols
    # Save the readme_df 
    readme_df.to_csv(filename)
    
def crawl_github(target: str, reparse=False, readme_cutoff=1000):
    '''Specify user to start a crawl and like to have run looking for
    user/repos, and README.md files
    '''
    # Establish users data that was crawled and if it was parsed or not
    parsed_users_file = 'users_data_c.csv'
    
    # Try to fetch readme_data_c to see how many entries there are to check against cutoff
    try:
        # Check the readme length
        readme_len = len(pd.read_csv('readme_data_c.csv', index_col=['user_repo']))
        
    except:
        # if the file does not exist set the readme length to 0
        readme_len = 0
    
    # Show the number of README Destinations
    print('Current number of README destinations is:', readme_len)

    # Check if there is a file for that site
    if os.path.exists(parsed_users_file):
        # Pull the dataframe in
        df = pd.read_csv(parsed_users_file, index_col='user')
        # check if the target is in the dataframe
        if reparse:
            df.loc[target, 'parsed'] = False
        
        # Check if user is in the list
        elif target not in df.index.to_list():
            # Add target to the dataframe
            df.loc[target] = {'parsed': False}
        # Ensure the number of readme files is greater than the cutoff
        elif readme_len >= readme_cutoff:
            # If greater than the cutoff return the dataframe
            return df
            
    else:
        # If the dataframe does not exist, set the first value to the target
        # Set the user and set the user as the index
        df = pd.DataFrame([{'user': target, 'parsed': False}]).set_index('user')
            
    # Ensure there are no more users to parse
    while len(df[~df.parsed].parsed.to_list()) != 0 and readme_len < readme_cutoff:
        # Pull the dataframe in again to ensure it's fresh each iteration
        if os.path.exists(parsed_users_file):
        # Pull the dataframe in if it's not the first time
            if reparse:
                # Ensure the user is set to not parsed if reparse is True
                df.loc[target, 'parsed'] = False
                # Set reparse flag to False
                reparse = False
                
            # If not reparse and the user is not in the index
            elif target not in df.index.to_list():
                # Add user to the dataframe with parsed to False
                df.loc[target] = {'parsed': False}
            
        # Set the user to parse as the first element in the list of NON parsed users
        user = df[~df.parsed].index[0]

        # Returns a list of followers and a list of repositories
        followers, repos = fetch_user_followers_and_repos(user)
        
        # Have function check if any of the followers are in the dataframe already
        df = add_followers_to_dataframe(df, followers)
        
        # Send current user and first 30 repos to be added to readme_df
        add_repos_to_readme_dataframe(user, repos)
        
        # Get the number of readme entries and check against cutoff
        readme_len = len(pd.read_csv('readme_data_c.csv', index_col=['user_repo']))
        
        print('Current number of README destinations is:', readme_len)
        
        # Set the current user parsed to True to iterate to next user
        df.loc[user] = {'parsed': True}
    
        # Save the dataframe so that it can continue to go through and check each url
        df.to_csv(parsed_users_file)
        # Show that the user was successfully parsed
        print(f'Parsed {user}')
        
    # Return dataframe of followers when done
    return df

# ## Now that we have the README destinations, we need to pull the README text and the associated script langauge that is used

def extract_text_and_lang(url):
    ''' Takes a Github url repo and pulls out the text and the associated
    programming language and percentage and returns a string of text and
    and a dictonary of languages formatted as a string where structure is 
    like this 
    '[{'programming_language': '__some_language__',
        'percentage': float(__some_percentage__)}]'
    '''
    # Set headers for BeautifulSoup request
    headers= {'User-Agent': 'Codeup Data Science'}
    
    # Fetch response from url
    response = get(url, headers=headers)
    # Create Soup html object
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # arguments be passed to the soup find_all function for text
    kwargs = {'class': 'markdown-body entry-content container-lg'}
    # returns the text from the soup object
    text_html = soup.find_all(**kwargs)
    # Extracted text from the readme file
    text_list = re.findall(r'>(.*?)<', str(text_html))
    
    # Filter out all blank strings and spaces and concat all together
    text = ' '.join([txt for txt in text_list if txt not in ['', ' ']])
    
    # arguments be passed to the soup find_all function for language
    kwargs = {
        'data-ga-click': 'Repository, language stats search click, location:repo overview'
    }
    # Find the raw languange html from url
    lang_html = soup.find_all(**kwargs)
    # regex out all the text from the html and remove
    # all the undesired results
    l_list = [txt for txt in re.findall(
        r'>(.*?)<', str(lang_html)) if txt not in ['', ' ', ', ']]
    
    # Iterate though the list and put the language and percentage into a 
    # list of dicts
    languages = [{'programming_language': l_list[n],
             'percentage': float(l_list[n+1][:-1])} for n in range(0, len(l_list), 2)]
    
    # Returns the text and the languages as strings
    return text, str(languages)


def parse_readme_destinations(cutoff=10000, reparse=False, n=0):
    '''Iterate though and parse README destinations untill we reach the cutoff limiter from the
    readme_data_c file. The default cutoff is 1000 and reparse default to False and n is the
    starting point for a reparse and will not work unless it's marked to True
    '''
    # Set the filename for the readme
    filename = 'readme_data_c.csv'
    
    # Checks to see if the readme exists
    if os.path.exists(filename):
        # pull the readme_df in and set the index col to the user_repo and
        # Reset the index so that the user_repo won't have issues being pulled out
        readme_df = pd.read_csv(filename, index_col='user_repo').reset_index()
    
    # If the readme_df does not exist print that it does not exist and run github_crawl first
    else:
        print(f'{filename} does not exist, run github_crawl first to build the file')
        return None
    
    # If reparse is requested set the readme_len to 0 so it will iterates
    if reparse:
        readme_len = 0
        
    # Otherwise calculate the len of the readme file
    else: 
        # Calculate the number of parsed destinations
        readme_len = len(readme_df[readme_df.parsed])
    
    # Check how many destinations were successfully parsed and if 
    # it is greater or equal to the cutoff return the dataframe
    if readme_len >= cutoff:
        # Return readme dataframe
        return readme_df
    
    # Continue parsing until the number of parsed values is greater than or equal to
    # the defined cutoff
    while readme_len < cutoff:
        # pull the readme_df in and set the index col to the user_repo and
        # Reset the index so that the user_repo won't have issues being pulled out
        readme_df = pd.read_csv(filename, index_col='user_repo').reset_index()
                
        # If reparse is desired just select from 
        if reparse:
            # Select the user_repo destination from readme_df and input into the url
            # Select the current readme to update later
            current_readme = readme_df.iloc[n]
            # Fetch the user_repo string
            user_repo = current_readme.user_repo
            # Build the url to the repo
            url = f'https://github.com/{user_repo}'
            # Increase n to get the next one next time
            n += 1
        else:
            # Select the first destination and build the URL target
            # from the non-parsed values if reparse is not requested
            # Select the user_repo destination from readme_df and input into the url
            # Select current readme to update later
            # Select the first index that is False
            n = (readme_df.parsed == False).idxmax()
            current_readme = readme_df.loc[n]
            # Fetch the user_repo string
            user_repo = current_readme.user_repo
            # Build the url the repo
            url = f'https://github.com/{current_readme.user_repo}'
            
        # Send the url to have the text and language be parsed and extracted
        text, lang= extract_text_and_lang(url)
        
        # These are going to be the values that are entered into the 
        # dataframe
        new_values = [
            user_repo, # The user_repo name
            True, # Set the value to True
            text, # The text combined that was queried
            lang # string of programming languages 
        ]
        # Check if the there is not any text in the readme
        if text == '':
            readme_df = readme_df.drop(n)
        # otherwise update with new values
        else:
            # Update the current readme index with the values
            readme_df.loc[n] = new_values
        # update the readme_len
        readme_len = len(readme_df[readme_df.parsed])
        print(readme_len)
        
        # Set the index to the user_repo
        readme_df= readme_df.set_index('user_repo')
        # Save the file so it keeps progress
        readme_df.to_csv(filename)
        
    # Once cutoff is reached, return the readme_df
    return readme_df