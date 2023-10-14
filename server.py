
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
    "http://localhost:8001",
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

class Prod2VecRecommender():
    def __init__(self, min_count=2, size=100, window=5, decay_alpha=0.9, workers=4):
        super(Prod2VecRecommender, self).__init__()
        self.min_count = min_count
        self.size = size
        self.window = window
        self.decay_alpha = decay_alpha
        self.workers = workers

    def __str__(self):
        return 'Prod2VecRecommender(min_count={min_count}, ' \
               'size={size}, ' \
               'window={window}, ' \
               'decay_alpha={decay_alpha}, ' \
               'workers={workers})'.format(**self.__dict__)

    def fit(self, train_data):
        sequences = train_data['sequence']
        self.model = Word2Vec(sentences=sequences, vector_size=self.size, window=self.window, min_count=self.min_count, workers=self.workers, sg=0)
        self.model.train(sequences, total_examples=len(sequences), epochs=100)
        return self


prod2vecRecom = pickle.load(open('prod2vec.pkl', 'rb'))



new_transaction_table = pd.read_csv(r'./Recommeder/Data/new_transaction_table.csv')
for i in range(len(new_transaction_table)):
    new_transaction_table['sequence'][i] = new_transaction_table['sequence'][i].replace("[","").replace("]","").replace("'","").split(", ")



# # wordsinVocab
# word2vecAllembeddings = pd.read_csv(r'./embeddings.pkl')
# read embeddings.pkl
word2vecAllembeddings = pd.read_pickle('embeddings.pkl')


# print("User ID: ", userId)
pidToProductMetadata = json.load(open(r'./Recommeder/Data/pidToProductMetadata.json', 'r'))

products = pd.read_csv(r'./Recommeder/Data/flipkart_com-ecommerce_sample.csv')
pidToidx = pd.Series(products.index,index=products['uniq_id']).drop_duplicates()
# pidToProductMetadata = 
indices = pd.Series(products.index,index=products['product_name']).drop_duplicates()


def reccomendProductsw2vec(userId):

    topk = 10
    
    pastPurchases = new_transaction_table[new_transaction_table['UID'] == userId]['sequence'].values

    pastPurchasesIdx = []
    for purchase in pastPurchases:
        pastPurchasesIdx.append(pidToidx[purchase].to_list())
    pastPurchasesIdx = pastPurchasesIdx[0]
    # print(pastPurchasesIdx)
    product_embeddings = []
    for idx in pastPurchasesIdx:
        product_embeddings.append(word2vecAllembeddings.iloc[idx]['vector'])
    
    product_embeddings = np.array(product_embeddings)
    avg_embeddings = np.mean(product_embeddings , axis=0)
    similarity_scores = []
    for idx , row in word2vecAllembeddings.iterrows():
        score = cosine_similarity([avg_embeddings],[row['vector']])
        similarity_scores.append((idx,score[0][0]))
        
    
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    product_names = {}
    last_score = 0
    for i in similarity_scores:
        if last_score == 0 or abs(last_score - i[1]) >= 0.01*last_score:
            indx = i[0]
            pid = products['uniq_id'].iloc[indx]
            pidToProductMetadata[pid]['image'] = pidToProductMetadata[pid]['image'].strip("\"")
            product_names[pid] = (pidToProductMetadata[pid])
            last_score = i[1]
            if len(product_names) == topk:
                break
            
        
    return product_names


def getreccomendationProd2vec(userId ):
    reccomdations = []
    
    pastPurchases = new_transaction_table[new_transaction_table['UID'] == userId]['sequence']

    pastPurchases = pastPurchases.tolist()[0]
    # print(pastPurchases , type(pastPurchases))
    reccomdations = prod2vecRecom.model.wv.most_similar(positive=pastPurchases[0], topn=10)

    reccomdation = {}
    for i in reccomdations:
        pidToProductMetadata[i[0]]['image'] = pidToProductMetadata[i[0]]['image'].strip("\"")
        reccomdation[i[0]] = pidToProductMetadata[i[0]]
    return reccomdation


@app.get("/Product2vec/{userId}")
def get_reccomendation(userId: str):
    return getreccomendationProd2vec(userId)

@app.get("/Word2vec/{userId}")
def getWord2reccomendation(userId: str):
    return reccomendProductsw2vec(userId)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)