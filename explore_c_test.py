import re
import unicodedata
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import nltk.sentiment

from wordcloud import WordCloud

plt.rc('figure', figsize=(13, 7))
plt.style.use('seaborn-darkgrid')

from prepare_c import *

class NLP_explore():
    '''Explores some plots and other features and frequencies, bigrams of 
    two selected comparisons within a dataframe based on a specific label
    '''
    def __init__(self, data: pd.DataFrame, label_col:str, text_col:str, target:str):
        '''Pass df, label_col (what is the category), text_col is
        where the text is located, and target & not_target are the comparison strings
        '''
        df = data.copy(deep=True)
        # Create all class attributes
        self.label_col = label_col
        self.text_col = text_col
        # Create new instance from orginial
        self.df = data.copy()
        # Comparison 1
        self.target = target
        # Comparison 2
        self.not_target = f'not_{target}'
        
        # Clean the text
        # This is where you'd apply the cleaning for all data
        self.df[text_col] = self.df[text_col].apply(basic_clean)
        
        #TODO change the comparison to not comp and comp
        # Select comparison 1 from the labeled column
        self.comp_series_1 = self.df[self.df[label_col] == target][text_col]
        # Select comparison 2 from the labeled column
        self.comp_series_2 = self.df[self.df[label_col] != target][text_col]
        # Joined text from comparison 1
        self.comp_text_1 = self.comp_series_1.str.cat(sep=' ')
        # Joined text from comparison 2
        self.comp_text_2 = self.comp_series_2.str.cat(sep=' ')
        # All cleaned words combined
        self.all_words = f'{self.comp_text_1} {self.comp_text_2}'        
        
        # Calculates cond1 frequencies
        self.cond1_freq = pd.Series(self.comp_text_1.split()).value_counts()
        # Calculates cond2 frequencies
        self.cond2_freq = pd.Series(self.comp_text_2.split()).value_counts()
        # Calculates all word frequencies
        self.all_freq = pd.Series(self.all_words.split()).value_counts()
        
        # Creates attribute of word_counts to view the word counts
        # for the condition
        word_counts = pd.concat(
            [self.cond1_freq,
             self.cond2_freq,
             self.all_freq],
            axis=1).fillna(0).astype(int)
        word_counts.columns=[target, not_target, 'all']
        # Make word_counts a feature
        self.word_counts = word_counts
                       
        
    # Create method to plot a horizontal plot for the frequencies
    def hplot_word_freq_viz(self, n=20, sort='all', asc=False):
        '''Takes n which is the number of top values you'd like to plot
        and a sort col, default is 'all', ascending defaults to False
        but you can also choose while one of the current labels you'd like to 
        use as a sorting feature
        '''
        plt.rc('font', size=18)
        self.word_counts.\
            sort_values(sort, ascending=asc
                       ).head(n)[[self.target, self.not_target]].plot.barh()
        plt.title(f'''{self.target.capitalize()} vs {self.not_target.capitalize()} count for the top {n} most frequent words''')
        plt.xlabel('Count')
        plt.show()
        
        return
        
    # Create method to stacked barplot for word counts
    def stacked_bplot_freq(self, n=20, sort='all', asc=False):
        '''Takes n which is the number of top values you'd like to plot
        and a sort col, default is 'all', ascending defaults to False
        plots a stacked barchart plot but you can also choose while one 
        of the current labels you'd like to use as a sorting feature
        '''
        plt.figure(figsize=(16, 9))
        plt.rc('font', size=16)

        (self.word_counts.sort_values(sort, ascending=asc)
         .head(n)
         .apply(lambda row: row/row['all'], axis = 1)
         .drop(columns = 'all')
         .sort_values(by = self.target)
         .plot.barh(stacked = True, width = 1, ec = 'k')
        )
        plt.title(f'''% of {self.target.capitalize()} vs {self.not_target.capitalize()} count for the top {n} most frequent words''')
        plt.xlabel('Percentage')
        plt.show()
        
        return
    
    # Create method to plot a horizontal plot for the bigrams
    def n_gram(self, col='all', n=2, top_n=20, asc=False):
        '''Takes n which is the number of top values you'd like to pull from the 
        bigram and then you plot those pairs on a horizontal plot 
        default is 'all', ascending defaults to False, but you can also choose
        while one of the current labels you'd like to use as a sorting feature
        '''
        # generates bigrams which are combinations of two words 
        # throughout the string
        
        # Condition 1 bigrams
        if col == self.target:
            grams = list(nltk.ngrams(self.comp_text_1.split(), n))
        # Condition 2 bigrams
        elif col == self.not_target:
            grams = list(nltk.ngrams(self.comp_text_2.split(), n))
        # If all or not in the conditions go to all
        else:
            grams = list(nltk.ngrams(self.all_words.split(), n))
            
        
        # Plot the bigrams
        pd.Series(grams).value_counts().head(top_n).plot.barh()
        # Set the title
        plt.title(f'Top {top_n} most common {col.title()} {n}_grams')
        # Label the x axis with count
        plt.xlabel('Count')
        # Show plot
        plt.show()
        
        # Return the bigram for that selected column
        return grams
    
    # Create wordcloud plot based on condition
    def plot_wordcloud(self, col='all', save=False):
        '''Allows for a wordcloud to be plotted from the 'col' of text
        but you can also choose while one of the current labels you'd like to 
        use as a sorting feature
        '''
        # Condition 1 words
        if col == self.target:
            words = self.comp_text_1
        # Condition 2 words
        elif col == self.not_target:
            words = self.comp_text_2
        # If all or not in the conditions go to all
        else:
            words = self.all_words
            
        img = WordCloud(background_color='white', width=800, height=600).generate(words)
        plt.imshow(img)
        plt.axis('off')
        # If save = True, then save the image
        if save:
                plt.savefig(f'wordcloud.png')
        
        return
    
    # Adds sentiment analysis to the instance dataframe
    def add_sentiment_analysis(self, cols=['neg', 'neu', 'pos', 'compound']):
        '''Calculates sentiment analysis scores for all columns by
        default, remove names as desired
        cols = ['neg', 'neu', 'pos', 'compound']
        '''
        # Create the Sentiment Analyzer
        sia = nltk.sentiment.SentimentIntensityAnalyzer()
        
        # Iterate though each row
        def fetch_sentiment(row):
            '''Takes a row from dataframe that has text and returns the SentimentIntensityAnalyzer 
            for that text
            '''
            # Stores sentiment results to check with desired cols
            sentiment_dict = sia.polarity_scores(row[self.text_col])
            
            # Returns only the desired columns from the dictonary
            return [sentiment_dict[i] for i in cols if i in sentiment_dict]
        
        # Apply and expand columns with the results from the sentiment analysis and selected columns
        self.df[cols] = self.df.apply(fetch_sentiment, axis=1, result_type='expand')
    
        return self.df
            

    def add_features(self, features=[
        'message_length',
        'word_count',
        'unique_word_count',
        'avg_word_len'
    ]):
        
        '''These are all the features that can be added to the dataframe
        default = [
        'message_length', word_count', 'unique_word_count','avg_word_len'
        ]
        '''
        
        def calculate_feature(row):
            '''Goes through the dict of values that need to be added 
            and puts them into a dict, and then returns it on a row-by-row
            basis
            '''
            # The dict to be queried at the end as a placeholder
            result = dict()
            
            # Tabulates all features then adds only the ones desired at
            # the end.
            txt = row[self.text_col] # Pulls the text out of the row
            
            # Put the text into a panda Series for easier tabulations
            s = pd.Series(txt.split(), dtype = 'object')
            
            # Tabulate message length via sum of the all text and chars
            result['message_length'] = len(txt)
            
            # Tabulate word count row wise
            result['word_count'] = s.size
            
            # Tabulate unique word count take unique value counts
            result['unique_word_count'] = s.value_counts().to_dict()
            
            # Tabulate avgerage word length per row by taking total sum, without whitespace / total word count
            try: # Need to ensure we do not divide by zero
                avg = result['message_length'] / result['word_count']
            except: # If divide by zero error, set avg to zero
                avg = 0
            result['avg_word_len'] = round(avg, 2)

            # Filter any features not desired
            return [result[i] for i in features if i in result]
        
        # Apply and expand columns with the results from the all_calculated features and selected columns
        self.df[features] = self.df.apply(calculate_feature, axis=1, result_type='expand')
        
        return self.df
    
    def sentiment_bivariate_plots(self, sentiments=['compound'],
                                  features=['message_length','word_count', 'avg_word_len'],
                                  save=False, alpha=0.5, levels=30):
        '''Takes sentiment columns as a list and feature columns as a list and plots each comb
        default sentiment is ['compound']
        default features are = [
        'message_length', word_count', 'avg_word_len']
        default alpha = 0.5
        default levels = 30
        default save = False

        '''
        comp_series_1 = self.target
        comp_series_2 = self.not_target
        comp_series_1_df = self.df[self.df[self.label_col] == comp_series_1]
        comp_series_2_df = self.df[self.df[self.label_col] == comp_series_2]
        # Iterate through features
        for feature in features:
            # Iterate though sentiments
            for sentiment in sentiments:
                filename = f'{feature}_vs_{sentiment}'
                
                
                # Bivariate KDE plot for Comparison 1 first
                sns.kdeplot(x= comp_series_1_df[feature], 
                            y= comp_series_1_df[sentiment],
                            levels = levels, shade = True, alpha=alpha, label=f'{comp_series_1.title()}'
                           ).set(title=f'{feature.title()} vs {sentiment.title()} for {comp_series_1}')
                # Then comparison 2
                sns.kdeplot(x= comp_series_2_df[feature], 
                            y= comp_series_2_df[sentiment],
                            levels = levels, shade = True, alpha=alpha, label=f'{comp_series_2.title()}'
                           ).set(title=f'{feature.title()} vs {sentiment.title()} for {comp_series_2}')
                plt.legend()
                plt.show()
                
                # If save = True, then save the image
                if save:
                    plt.savefig(filename+'_KDE.png')
                
                # Plot scatterplot
                sns.relplot(
                    data = self.df, x = feature,
                    y = sentiment, hue = self.label_col).set(title=f'{feature.title()} vs {sentiment.title()}')
                plt.legend()
                plt.show()
                
                # If save = True, then save the image
                if save:
                    plt.savefig(filename+'_scatter.png')
                
    def sentiment_distributions(self, sentiments=['neg', 'neu', 'pos', 'compound'], save=False):
        '''Takes sentiment columns as a list and feature columns as a list and plots each comb
        default sentiment is ['neg', 'neu', 'pos', 'compound']
        '''
        # Iterate though sentiments
        for sentiment in sentiments:
            comp_series_1_df = self.df[self.df[self.label_col] == self.target]
            comp_series_2_df = self.df[self.df[self.label_col] == self.not_target]

            # Plot Dist for Comparison 1 first
            sns.kdeplot(comp_series_1_df[sentiment], label = self.target.title())
            # Then plot Dist for Comp 2
            sns.kdeplot(comp_series_2_df[sentiment], label = self.not_target.title())
            plt.legend()
            plt.show()
            
            # If save = True, then save the image
            if save:
                plt.savefig(f'{sentiment}_dist.png')
        
        
        