# README

Our recommendation system offers personalized product suggestions by employing a variety of algorithms. This provides users with product recommendations based on their preferences and the purchase history of similar users.

## **Table of Contents**

- **Features**
- **Algorithms Implemented**
- **Use Cases**
- **Limitations**
- **Future Scope**

## **Features**

- Input a **`user_id`** to receive product recommendations based on similar user purchases and specific user preferences.
- Conduct a product search and get suggestions for related items.
- Recommendations based on user search history and product similarity.

## **Algorithms Implemented**

### **1. Collaborative Filtering**

- Leverages user interactions (like purchases and views) to generate personalized product recommendations.
- Provides insights into collective user preferences.
- Challenges include scalability with larger user groups.

### **2. Word2Vec**

- Recommends products by finding similar product embeddings based on previous user purchases.
- Highlights relationships between products for precise recommendations.

### **3. TF-IDF for Product Embeddings**

- Creates product embeddings to measure product similarity using cosine similarity.
- Focuses on product attributes like descriptions, tags, and categories.
- High dimensionality of the embeddings may introduce computational challenges.

### **4. Prod2Vec**

- A modified Word2Vec tailored for e-commerce.
- Products are treated as word tokens; user sequences are used to determine product embeddings.
- Understands the semantics of products, crucial for evaluating product similarity.

## **Use Cases**

1. **Real-time Homepage Recommendations:** Dynamically suggest personalized products on the homepage.
2. **In-App Search Enhancements:** Personalize search results based on user preferences.
3. **Cart Enhancement Recommendations:** Offer related products post-purchase to encourage repeat transactions.
4. **User Profile Recommendations:** Adapt recommendations using user profiles to boost relevance.

## **Limitations**

- **Synthetic Dataset:** Uses synthetic data due to the absence of real data, which may not capture genuine user behaviors.
- **Multiple Model Creation:** Using various models can complicate deployment and introduce inconsistencies.

## **Future Scope**

- **Hybrid Model Integration:** Develop a unified model by merging the strengths of individual models.
- **Scalability Enhancement:** Optimize the system for greater scalability.
- **Category-Based Recommendations:** Use advanced clustering techniques for recommendations.
- **Personalized Buying Capacity:** Recommend products based on user's buying capacity.
- **Real-World Data Utilization:** Use genuine user data to refine recommendation models.

## **Technologies Used**

- Python
- Jupyter Notebook (.ipynb)
- TF-IDF
- Word2Vec
- Prod2Vec
- Collaborative Filtering