#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import psycopg2
import json
import pandas as pd
RS_SECRET_NAME = 'prod-redshift-cluster-2'
RS_REGION = 'us-east-1'
def reconnect(f):
    def wrapper(client, *a, **kw):
        if client.conn.closed != 0:
            client.connect()
        return f(client, *a, **kw)
    return wrapper
class redshift_client:
    def __init__(self, secret_name=RS_SECRET_NAME, region_name=RS_REGION):
        self.secret_name = secret_name
        self.region_name = region_name
        self.connect()
    def connect(self):
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=self.region_name
        )
        secret = client.get_secret_value(
                 SecretId=self.secret_name
        )
        credentials = json.loads(secret['SecretString'])
        self.conn = psycopg2.connect(database='irl',
                         user=credentials['username'],
                         host=credentials['host'],
                         password=credentials['password'],
                         port=credentials['port'])
    @reconnect
    def query(self, query_str):
        return pd.read_sql_query(query_str, self.conn)


# In[2]:


rc = redshift_client()


# In[3]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dateutil import parser
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[4]:


def load_data(query): # pass in query, return invite data
    rc = redshift_client()
    df = rc.query(query)
    return df


# In[5]:


def preprocess_invites(df): # pass in data, return clean data 
    df['title'] = df['title'].str.lower()
    df['title'] = [re.sub(r"[-()\"#/@;:<>{}=~|.?!,]", "", str(x)) for x in df['title']]
    return df


# In[6]:


def preprocess_groups(df): # pass in data, return clean data 
    df['name'] = df['name'].str.lower()
    df['name'] = [re.sub(r"[-()\"#/@;:<>{}=~|.?!,]", "", str(x)) for x in df['name']]
    return df


# In[7]:


def tokenize(input):
    """
    returns tokenized version of text - split into smaller units 
    """
    return word_tokenize(input)

def remove_stop_words(input):
    """
    returns text without stop words
    """
    input = word_tokenize(input)
    return [word for word in input if word not in stopwords.words('english')]

def lemmatize(input):
    """
    lemmatizes input to group all forms of word together 
    """
    lemmatizer=WordNetLemmatizer()
    input_str=word_tokenize(input)
    new_words = []
    for word in input_str:
        new_words.append(lemmatizer.lemmatize(word))
    return ' '.join(new_words)

def nlp_pipeline(input):
    """
    calls all other functions together to perform NLP on a given text
    """
    return lemmatize(' '.join(remove_stop_words((input))))


# In[8]:


def find_tags(invite_title):
    """
    takes invite title as an input, and finds the four most important topics/tags
    using CountVectorizer and LDA
    
    """
    # prepare for nlp with preprocessing pipeline
    text = nlp_pipeline(invite_title)
    # embed on vector
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform([text])
    # chose 4 tags for now to represent each invite
    number_topics = 1
    number_words = 4
    # create and fit LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    
    # assign words as feature names for topics/tags 
    words = count_vectorizer.get_feature_names()

    # get topics/tags from model using topic word distribution from lda
    topics = [[words[i] for i in topic.argsort()[:-number_words - 1:-1]] for (topic_idx, topic) in enumerate(lda.components_)]
    topics = np.array(topics).ravel()

    return topics


# In[9]:


def get_user_embeddings(user_df):
    """
    takes user invites and creates word embeddings using BERT
    
    """
    
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    user_embeddings = sbert_model.encode(user_df['title'])
    
    return user_embeddings


# In[10]:


def get_group_embeddings(groups_df):
    """
    takes group titles and creates word embeddings using BERT
    
    """
    
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    group_embeddings = sbert_model.encode(groups_df['name'])
    
    return group_embeddings


# In[11]:


def get_similarity(user_embeddings, group_embeddings):
    """
    gets cosine similarity of user invite embeddings and public group name embeddings
    
    """
    similarities = cosine_similarity(user_embeddings, group_embeddings)
    
    return similarities


# In[12]:


def recommend_group(similarities):
    """
    uses sort function to find most similar group based off of cosine similarity
    
    """
    group_rec = np.argsort(-similarities)[:,0]
    
    return group_rec


# In[13]:


def interpret_group(groups_df, group_rec):
    """
    uses group_rec to output group name recommended for each invite title 
    
    """
    
    group_name = np.array(groups_df['name'])[group_rec]


# In[14]:


def main():
    user_query = '''
    SELECT i.user_id, i.title
    FROM invites i
    INNER JOIN invites_users iu ON i.id = iu.invite_id
    WHERE i.is_public = 1 AND (iu.interested OR iu.attending)
    GROUP BY i.user_id, i.title, i.id, iu.invite_id
    '''
    user_df = load_data(user_query)
    
    groups_query = '''
    SELECT name, description
    FROM groups 
    WHERE is_public = 1 AND is_approved_for_explore = 1
    '''
    groups_df = load_data(groups_query)
    
    user_df = preprocess_invites(user_df)
    groups_df = preprocess_groups(groups_df)
    
    user_tags = user_df["title"].apply(find_tags).values
    groups_tags = groups_df["description"].apply(find_tags).values
    
    user_df['tags'] = user_tags
    groups_df['tags'] = groups_tags
    
    user_embeddings = get_user_embeddings(user_df)
    group_embeddings = get_group_embeddings(groups_df)
    
    similarities = get_similarity(user_embeddings,group_embeddings)
    
    group_rec = recommend_group(similarities)
    
    return group_rec, interpret_group(groups_df,group_rec)


# In[15]:


if __name__ == '__main__':
    main()

