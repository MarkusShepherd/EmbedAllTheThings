# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Data
# The data in this notebook can be downloaded from that Kaggle webpage at: https://www.kaggle.com/rdoume/beerreviews
# It requires Kaggle login so the download can not be included in this notebook.
#
# As of November 27, 2019 this notebook depends on functionality in the 0.4dev branch of umap.  To install that in your python environment use the following command:<BR>
# `pip install datashader holoviews`<BR>
# `pip install git+https://github.com/lmcinnes/umap.git@0.4dev`
#     
# At some point in the future this will be merged into the main branch and this notebook will only require:<BR>
# `pip install umap-learn`
#

# %%
import pandas as pd

# import numpy as np
import umap
import umap.plot
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Some plotting libraries
import matplotlib.pyplot as plt

# %matplotlib notebook
from bokeh.plotting import show, save, output_notebook, output_file
from bokeh.resources import INLINE

output_notebook(resources=INLINE)
# %load_ext nb_black
# %load_ext lab_black

# %% [markdown]
# We'd like to turn categorical data into a document of space seperated strings.  We want to do this to keep a nice easy pipeline for sklearns CountVectorizer.  A very natural way to accomplish this is via pandas df.groupby() function with a " ".join(my_array) aggregator passed in.  Unfortunately, it turns out that " ".join(my_array) seems to have trouble on for lists (or sequences) longer than 3,000 or so.  
#
# As such we've included a simple (though not necessarily efficient) join function that scales to large arrays.

# %%
def join(iterator, seperator):
    """
    This function casts the elements of iterator to strings then merges those strings together with a string 
    representation of seperator.  
    Had to write a custom join to handle very, very long lists of things. "".join falls appart above 3013.
    params
    iterator: an iterator.  This function makes use of the overload + operator for strings
    seperator: an item of the same class as is contained in our iterator to be added between every pair of instances.
    returns
    The sum of the iterator values with seperator iterposed between each.
    """
    it = map(str, iterator)
    seperator = str(seperator)
    string = next(it, "")
    for s in it:
        string += seperator + s
    return string


# %% [markdown]
# ### Read in our data

# %%
reviews = pd.read_csv("beerreviews_kaggle.zip")
reviews.shape

# %% [markdown]
# ### Interesting Tidbits:
# 25% of the beer_beerid have no recorded beer_abv.

# %%
reviews.head().T

# %% [markdown]
# ### What if we wanted to only group beer by the users that liked them?
#
# Are two beers similar if two reviewer tried them?  Perhaps not, instead lets filter to only the reviewers who enjoyed the beer.
#
# Because this is talking about reviewers and not beer we need to filter our initial data frame and re-run our process.

# %%
reviews.review_overall.value_counts().sort_index().plot.bar()

# %% [markdown]
# Wow, people like the beer that they try.  Given this plot, let's take a positive review to be 4.5 or higher.  We'll overwrite our variable (which is generally bad) in order to save on code reuse from the last notebook.

# %%
reviews = reviews[reviews.review_overall >= 4.5]
reviews.shape

# %% [markdown]
# ## Embed Beer
#
# If we are going to embed beer then we need to turn our reviews data frame into a frame with one row per beer instead of one row per review.
#
# This is a job for groupby.  We groupby the column we'd like to embedd and then use agg with a dictionary of column names to aggregation functions to tell it how to summarize the many reviews about a single beer into one record.  Aggregation functions are pretty much any function that takes an iterable and returns a single value.  Median and max are great functions for dealing with numeric fields.  First is handy for a field that you know to be common across for every beer review.  In other words fields that are tied to the beer such as brewery_name or beer_abv.
#
# If we've written a short lambda function to pack a list of categorical values into a space seperated string for later consumption by CountVectorizer.

# %%
# %%time
unique_join = lambda x: join(x.unique(), " ")
beer = reviews.groupby('beer_beerid').agg({
    'beer_name':'first',
    'brewery_name':'first',
    'beer_style':'first',
    'beer_abv':'mean',
    'review_aroma':'mean',
    'review_appearance':'mean',
    'review_overall':'mean',
    'review_palate':'mean',
    'review_taste':'mean',
    'review_profilename':[unique_join, len]
}).reset_index()

beer.columns = """beer_beerid beer_name brewery_name beer_style beer_abv 
review_aroma review_appearance review_overall review_palate review_taste 
review_profilename_list review_profilename_len""".split()
beer.shape

# %%
beer.head(3).T

# %% [markdown]
# ## Embed the data
#
# We are going to vectorize our data and look at the number of categorical values they have in common.  A useful thing to do here is to require each row to have a minimum support before being included.  Filtering this early, will ensure indices line up later on.

# %%
popular_beer = beer[beer.review_profilename_len > 50].reset_index(drop=True)

# %% [markdown]
# This step turns a sequence of space seperated text into a sparse matrix of counts.  One row per row of our data frame and one column per unique token that appeared in our categorical field of interest.
#
# If we want to deal with sets (i.e. just presence or absence of a category) use:<BR>
# `beer_by_authors_vectorizer = CountVectorizer(binary=True)`<BR>
# If we think counts should matter we might use:<BR>
# `beer_by_authors_vectorizer = CountVectorizer()`<BR>
# or if we want to correct for very unbalanced column frequencies:<BR>
# `beer_by_authors_vectorizer = TfidfVectorizer()`<BR>
#
#

# %%
beer_by_authors_vectorizer = CountVectorizer(binary=True, min_df=10)
beer_by_authors = beer_by_authors_vectorizer.fit_transform(
    popular_beer.review_profilename_list
)
beer_by_authors

# %% [markdown]
# Now we reduce the dimension of this data.
#
# If we are dealing with sets (i.e. just presence or absence of a category) use:<BR>
# `metric='jaccard'`<BR>
# If we think counts should matter we might use:<BR>
# `metric='hellinger'`<BR>
# or if we want to correct for very unbalanced column frequencies:<BR>
# `metric='hellinger'`<BR>
#     
# As you get more and more points I'd recommend increasing the `n_neighbors` parameter to compensate.  Thing of this as a resolution parameter.
#
# `n_components` controls the dimension you will be embedding your data into (2-dimensions for easy visualization).  Feel free to embed into higher dimensions for clustering if you'd like.
#
# `unique=True` says that if you have two identical points you want to map them to the exact same co-ordinates in your low space.  This becomes especially important if you have more exact dupes that your `n_neighbors` parameter.  That is the problem case where exact dupes can be pushed into very different regions of your space.

# %%
# %%time
beer_by_authors_model = umap.UMAP(n_neighbors=15, n_components=2, metric='jaccard', unique=True, random_state=42).fit(beer_by_authors.todense())

# %%
umap_plot = umap.plot.points(
    beer_by_authors_model, labels=popular_beer.brewery_name, theme="fire"
)
# umap_plot.figure.savefig('results/popular_beer_by_positive_reviewer_jaccard.png', dpi=300, bbox_inches='tight', transparent=True)

# %% [markdown]
# ... and now for an interactive plot with mouseover.

# %%
abv_label = popular_beer.beer_abv.fillna(0)
hover_df = popular_beer["beer_beerid beer_name brewery_name beer_style".split()]
f = umap.plot.interactive(
    beer_by_authors_model,
    labels=popular_beer.brewery_name,
    hover_data=hover_df,
    theme="fire",
    point_size=5,
)
# save(f,'results/popular_beer_by_positive_reviewers_jaccard.html')
show(f)
