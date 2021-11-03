<center><h1>Natural Language Processing - Github Programming Language Prediction</center>

<a name ='toc'></a>
# Table of Contents 
1. [Project Planning](#project_planning)
    1. [Project Objectives](#project_objectives)
    2. [Business Goals](#business_goals)
    3. [Audience](#audience)
    4. [Deliverables](#deliverables)
2. [Executive Summary](#exe_sum)
    1. [Goals](#goals)
    2. [Findings](#findings)
3. [Acquire Data](#acquire)
    1. [Data Dictonary](#data_dict)
    2. [Acquire Takeaways](#acquire_takeaways)
4. [Prepare Data](#prep_data)
    1. [Distributions](#distributions)
    2. [Prepare Takeaways](#prepare_takeaways)
5. [Data Exploration](#explore)
    1. [Correlations](#correlations)
    2. [Pairplot](#pairplot)
    3. [Explore Takeaways](#explore_takeaways)
6. [Hypothesis](#hypothesis)
    1. [Conclusion](#hyp_conclusion)
7. [Modeling & Evaluation](#modeling)
    1. [Term Frequency](#term_freq)
    2. [Inverse Document Frequency](#inverse_doc_freq)
8. [Project Delivery](#delivery)
    1. [Presentation](#presentation)

<hr style="border-top: 10px groove tan; margin-top: 5px; margin-bottom: 5px"></hr>

<a name='project_planning'></a>
## Project Planning
✓ 🟢 **Plan** ➜ ☐ _Acquire_ ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

<a name='project_objectives'></a>
### Project Objectives 
> - For this project our team is to build a model that can predict what programming language a repository will be, given the text of a README file.
> - In addition to this, we are to build a well-documented jupyter notebook that contains the analysis of this prediction.
> - Any abstracted modules that are created to make the presentation more clean, during the acquistion and preparation of data.
> - Finally, we are to build a few Google slides to present toward a general audience that summarizes the findings within the project, with many visualizations.

<a name='business_goals'></a>
### Business Goals 
> - Build a script that will find, then scrape the README files from Github.
> - Prepare, explore and clean the data so that it can be input into modeling.
> - Utilizie Term Frequence (TF) Inverse Document Frequency (IDF) and a combination of the two features to assist with the modeling.
> - Document all these steps thoroughly.

<a name='audience'></a>
### Audience 
> - General population and individuals without specific knowledge or understanding of the topic or subject.

<a name='deliverables'></a>
### Deliverables
> - A clearly named final notebook. This notebook will contain more detailed processes other than noted within the README and have abstracted scripts to assist on readability.
> - A README that explains what the project is, how to reproduce the project, and notes about the project.
> - A Python module or modules that automate the data acquisition and preparation process. These modules should be imported and used in your final notebook.

<div style="text-align: right"><a href='#toc'>Table of Contents</a></div>
<hr style="border-top: 10px groove tan; margin-top: 1px; margin-bottom: 1px"></hr>

<a name='exe_sum'></a>
## Executive Summary
> - 30,000 repository README files were scraped from Github.
> - Data was analyzed for the Python, Javascript, Java, and HTML programming languages.
> - Our model performed well, with an accuracy of over 90% __UPDATE LATER__

<a name='goals'></a>
### Goals
> - Build a model that can predict what programming language a repository will be, given the text of a README file.

<a name='findings'></a>
### Findings
> - What sets the READMEs apart? __UPDATE LATER__

<div style="text-align: right"><a href='#toc'>Table of Contents</a></div>
<hr style="border-top: 10px groove tan; margin-top: 1px; margin-bottom: 1px"></hr>

<a name='acquire'></a>
## Acquire Data
✓ _Plan_ ➜ 🟢 **Acquire** ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_
> - Our first issue was locating a sufficient number of README destinations to actually parse. The solution we devised was to look at a person's followers on Github, then add those followers to a list.
We would also parse up to the first 30 repository destinations from that user. Then we would iterate to the next follower and continue until we had around 30,000 repository destinations.
> - Once we had our destinations, we scraped the README text and all the programming languages and their associated percentages.
> - To determine the primary programming language of any repository, we first read in the percentages of the programming languages used in it and set a percentage threshold. So, if a programming language was at or above that threshold, it was considered the primary programming language of the repository.

### Total Missing Values
> - 15

<a name='data_dict'></a>
### DataFrame Dict

| Feature           | Datatype                         | Definition                                                 |
|:------------------|:---------------------------------|:-----------------------------------------------------------|
| readme            | 13777 non-null: object           | Original readme content of the scraped repository          |
| prog_lang         | 13792 non-null: object           | The predominant programming language used in the repository|
| original          | 13792 non-null: object           | Original readme content of the scraped repository          |
| stemmed           | 13777 non-null: object           | The cleaned, stemmed version of the readme                 |
| lemmatized        | 13777 non-null: object           | The cleaned, lemmatized version of the readme              |

<a name='acquire_takeaways'></a>
### Takeaways from Acquire:
> - Target Variable: prog_lang
> - This dataframe currently has 13,792 rows and 5 columns.
> - There are 15 missing values.
> - All columns are string object types.

<div style="text-align: right"><a href='#toc'>Table of Contents</a></div>
<hr style="border-top: 10px groove tan; margin-top: 1px; margin-bottom: 1px"></hr>

<a name='prep_data'></a>
## Prepare Data
✓ _Plan_ ➜ ✓ _Acquire_ ➜ 🟢 **Prepare** ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_
> - To clean the data, we:
    - Converted all characters to lowercase, 
    - Normalized unicode characters, 
    - Encoded into ascii byte strings and ignored unknown characters,
    - Decoded into usable UTF-8 strings,
    - Removed anything that was not either a letter, number, or whitespace,
    - tokenized the data.
> - To make our model more accurate, we decided to filter out words that that didn't seem to be important for identification purposes. To do this, we found the counts of each word used in the repositories for a particular programming language. Then, we removed any words that had a Z-score of .5 or below. This removed any junk words that may have been present and placed greater emphasis on the words that were most prevalent.
> - We performed this process twice. Once for the designated programming language, and then again for all of the repositories that were NOT that language.
> - Then we created both stemmed and lemmatized versions of the cleaned data.
> - Finally, we split the data into train and test sets.

<a name='prepare_takeaways'></a>
### Prepare Takeaways
> - 
> - 
                     
<div style="text-align: right"><a href='#toc'>Table of Contents</a></div>
<hr style="border-top: 10px groove tan; margin-top: 1px; margin-bottom: 1px"></hr>


<a name='explore'></a>
## Explore Data
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ 🟢 **Explore** ➜ ☐ _Model_ ➜ ☐ _Deliver_
> - 
> - 

<a name='correlations'></a>
### Correlations


#### Correlation Heatmap


#### Correlations Table


<a name='pairplot'></a>
### Pair Plot


<a name='explore_takeaways'></a>
### Explore Takeaways
> - 
> - 

<div style="text-align: right"><a href='#toc'>Table of Contents</a></div>
<hr style="border-top: 10px groove tan; margin-top: 1px; margin-bottom: 1px"></hr>

<a name='hypothesis'></a>
### Hypothesis 1 
> - 
> -

<a name='modeling'></a>
## Modeling & Evaluation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ 🟢 **Model** ➜ ☐ _Deliver_
> - 
> - 

<a name='term_freq'></a>
### Term Frequency - TF
> - 
> - 

<a name='inverse_doc_freq'></a>
### Inverse Document Frequency - IDF
> - 
> - 

<div style="text-align: right"><a href='#toc'>Table of Contents</a></div>
<hr style="border-top: 10px groove tan; margin-top: 1px; margin-bottom: 1px"></hr>

<a name='delivery'></a>
## Project Delivery
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ ✓ _Model_ ➜ 🟢 **Deliver**
> - 
> - 

<a name='presentation'></a>
### Presentation
> - 


<div style="text-align: right"><a href='#toc'>Table of Contents</a></div>
<hr style="border-top: 10px groove tan; margin-top: 1px; margin-bottom: 1px"></hr>