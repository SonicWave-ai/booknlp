import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download necessary NLTK data (for tokenization)
nltk.download('punkt')

# Load the book data (CSV)
data = pd.read_csv("books.csv", delimiter=";", header=None, 
                   names=["ISBN", "Title", "Author", "Year", "Publisher", "ThumbImage", "ImageURL1", "ImageURL2"])

# Function to get book recommendations
def recommend_books(query, data):
    # Create a TF-IDF Vectorizer to convert book titles to vectors
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data["Title"])

    # Transform the user's query into the same vector space
    query_vec = tfidf_vectorizer.transform([query])

    # Calculate the cosine similarity between the query and all book titles
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix)

    # Get the index of the most similar book
    similar_books_idx = np.argsort(cosine_sim[0])[::-1]

    # Display the top 5 most similar books
    recommendations = []
    for idx in similar_books_idx[:5]:
        recommendations.append({
            'Title': data["Title"][idx],
            'Author': data["Author"][idx],
            'Year': data["Year"][idx],
            'Publisher': data["Publisher"][idx],
            'Image': data["ImageURL1"][idx]
        })
    
    return recommendations

# Function to chat with the user and recommend books
def chatbot():
    print("Hello! I'm your Book Recommendation Bot.")
    print("Ask me about books by their title or author.")
    
    while True:
        query = input("\nWhat book are you looking for? (Type 'exit' to quit): ")
        
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        recommendations = recommend_books(query, data)
        
        print("\nHere are some book recommendations for you:")
        for i, book in enumerate(recommendations, start=1):
            print(f"{i}. {book['Title']} by {book['Author']} ({book['Year']})")
            print(f"Publisher: {book['Publisher']}")
            print(f"Cover Image: {book['Image']}")
            print("-" * 50)

# Run the chatbot
if __name__ == "__main__":
    chatbot()
