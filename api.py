import numpy as np
from fastapi import FastAPI
import pickle
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity

# Load user_mapping from the file
with open('user_mapping.pkl', 'rb') as file:
    user_mapping = pickle.load(file)

with open('product_mapping.pkl', 'rb') as file:
    product_mapping = pickle.load(file)

# Load saved user and product embeddings
user_embeddings = np.load('user_embeddings.npy')
product_embeddings = np.load('product_embeddings.npy')
reverse_product_mapping = {idx: product_id for product_id, idx in product_mapping.items()}


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


@app.get("/recommend/{user_id}")
def recommend(user_id: str):
    if user_id in user_mapping:
        user_index = user_mapping[user_id]
        recommendations = []

        for product_idx in range(len(product_embeddings)):
            predicted_rating = np.dot(user_embeddings[user_index], product_embeddings[product_idx])
            recommendations.append((reverse_product_mapping[product_idx], predicted_rating))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        top_recommendations = recommendations[:10]
        rec = []
        for r in top_recommendations:
            dic = {"id":r[0], "url":"https://images.amazon.com/images/P/"+r[0]+".01._SCLZZZZZZZ_.jpg"}
            rec.append(dic)
        
        return rec
    else:
        return {"error": "User ID not found"}


if __name__ == "__main__":
    import uvicorn
    
    # Run your FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
