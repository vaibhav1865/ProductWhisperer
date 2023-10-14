
import pickle
import pandas as pd
import numpy as np
# import fastapi
from fastapi import FastAPI
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from fastapi.middleware.cors import CORSMiddleware



import json

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://192.168.208.20:8000" # Add any other allowed origins
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


sig = np.load('sig.npy')
tfv_matrix = pickle.load(open('tfv_matrix.pkl', 'rb'))
products = pd.read_csv(r'./Recommeder/Data/flipkart_com-ecommerce_sample.csv')

indices = pd.Series(products.index,index=products['product_name']).drop_duplicates()

tfv = pickle.load(open('tfv.pkl', 'rb'))
tfv_matrix = pickle.load(open('tfv_matrix.pkl', 'rb'))

new_transaction_table = pd.read_csv(r'./Recommeder/Data/new_transaction_table.csv')
for i in range(len(new_transaction_table)):
    new_transaction_table['sequence'][i] = new_transaction_table['sequence'][i].replace("[","").replace("]","").replace("'","").split(", ")

pidToProductMetadata = json.load(open(r'./Recommeder/Data/pidToProductMetadata.json', 'r'))

def getNearestProduct(unknownWord , topk=10 ):
    newSig = sigmoid_kernel(tfv_matrix , tfv.transform([unknownWord]))
    # idx = np.argsort(
    sigScore = list(enumerate(newSig))
    sigScore = sorted(sigScore , key=lambda x:x[1] , reverse=True)
    sigScore = sigScore[0:topk]
    idx = [i[0] for i in sigScore]

    # print(newSig.shape ,newSig)

    prodcuts = {}
    for i in idx:
        pid = products['uniq_id'].iloc[i]
        pidToProductMetadata[pid]['image'] = pidToProductMetadata[pid]['image'].strip("\"")
        prodcuts[pid] = pidToProductMetadata[pid]
    return prodcuts

def getPreviousPurchases(userID):
    pids =  new_transaction_table[new_transaction_table['UID'] == userID]['sequence'].values[0]
    products = {}
    for pid in pids:
        products[pid] = pidToProductMetadata[pid]
    return products




def tfidfSim(title):
    topk = 10
    if title not in products['product_name'].unique():
        return getNearestProduct(title , topk)
    
    indx = indices[title]
    
    sig_scores = []
    for i in range(len(sig[indx])):
        sig_scores.append((i,sig[indx][i]))
    sig_scores = sorted(sig_scores , key=lambda x:x[1] , reverse=True)

    
    sig_scores = sig_scores[1:topk+1]
    
    product_indices = [i[0] for i in sig_scores]
    
    prodcuts = {}
    for i in product_indices:
        pid = products['uniq_id'].iloc[i]
        pidToProductMetadata[pid]['image'] = pidToProductMetadata[pid]['image'].strip("\"")
        prodcuts[pid] = pidToProductMetadata[pid]
    return prodcuts
    

@app.get("/tfid/{product}")
def getreccomendation(product: str):
    return tfidfSim(product)

if __name__ == "__main__":
    import uvicorn
    
    # Run your FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8002)