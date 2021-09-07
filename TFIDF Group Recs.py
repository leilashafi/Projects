#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[5]:


import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle 
import numpy as np

def load_data(query): # pass in query, return invite data
    rc = redshift_client()
    df = rc.query(query)
    return df

def preprocess(df): # pass in data, return clean data 
    df['title'] = df['title'].str.lower()
    df['title'] = [re.sub(r"[-()\"#/@;:<>{}=~|.?!,]", "", str(x)) for x in df['title']]
    return df

def get_tfidf_embeddings(df): # pass in data, return embedding matrix
    documents = df['title'].values.astype("U")
    vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.5, stop_words = "english")
    features = vectorizer.fit_transform(documents)
    
    # Save the vectorizer
    vec_file = 'vectorizer.pickle'
    pickle.dump(vectorizer, open(vec_file, 'wb'))
    
    return features

def fit_clusters(features): # pass in tfidf matrix, return model
    random_state = 0
    model = MiniBatchKMeans(n_clusters=25, random_state=random_state)
    model.fit(features) 

    # Save the model
    mod_file = 'kmeans.model'
    pickle.dump(model, open(mod_file, 'wb'))
   
    return model

def predict_clusters(features):  # pass in tfidf matrix, return cluster prediction 
    model = fit_clusters(features)
    prediction = model.predict(features)
    return prediction


def get_public_groups(public_groups_query): # pass in query return public groups data 
    rc = redshift_client()
    public_groups = rc.query(public_groups_query)
    return public_groups

def assign_cluster_to_public_group(public_groups): # pass in data + vars 
    docs = public_groups['description'].values.astype("U")  # return group assignment
    
    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
    loaded_model = pickle.load(open('kmeans.model', 'rb'))
    
    public_group_embeddings = loaded_vectorizer.transform(docs)
    cluster_centroids = loaded_model.cluster_centers_
    vec_file = 'cluster_centroids.pickle'
    pickle.dump(cluster_centroids, open(vec_file, 'wb'))
    
    
    c_s = cosine_similarity(public_group_embeddings, cluster_centroids)
    rec = np.argsort(-c_s)[:,0]
    
    rec_file = 'rec.pickle'
    pickle.dump(rec, open(rec_file, 'wb'))
    
    return rec

def interpret_group(rec): # pass in rec, return interpretation
    
    # for the cluster, print feature words
    loaded_rec = pickle.load(open('rec.pickle', 'rb'))
    loaded_model = pickle.load(open('kmeans.model', 'rb'))
    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
    arr = np.argsort(-loaded_model.cluster_centers_[loaded_rec[0:3]])
    
    feature_names = loaded_vectorizer.get_feature_names() 
    b = arr[[0,1,2]]
    labels = np.array(feature_names)[b]
    return labels

def main():
    
    query = 'SELECT user_id, title FROM invites WHERE is_public = 1 AND date >= cast(getdate() as date)'
    data = load_data(query)
    
    data_frame = preprocess(data)
    features = get_tfidf_embeddings(data_frame)
    model = fit_clusters(features) # necessary?
    prediction = predict_clusters(features)
    
    public_groups_query = 'SELECT name, description FROM groups WHERE is_public = 1 AND is_approved_for_explore = 1'
    public_groups = get_public_groups(public_groups_query)
    
    rec = assign_cluster_to_public_group(public_groups)
    return rec, interpret_group(rec)
                                                                            

if __name__ == '__main__':
    #main script goes here
    #combine above 4 functions
    main()
    


# In[ ]:




