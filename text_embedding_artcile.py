import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np

# Database configuration
DATABASE_URI = "postgresql://edb_admin:XXXX@xxxxx.pg.biganimal.io/blog_data"

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_text_data():
    """Fetches text data from the database."""
    with psycopg2.connect(DATABASE_URI) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, title, body  FROM articles  WHERE embedding  IS NULL")
            data = cursor.fetchall()
    return data

def generate_embeddings(text1, text2):
    """Generates embeddings for the combined text using the all-MiniLM-L6-v2 model."""
    combined_text = text1 + " " + text2  # Combine the text from both columns
    embedding = model.encode(combined_text)
    # Normalize the embedding to have unit length
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

def update_database_with_embeddings(data):
    """Updates the database with embeddings."""
    with psycopg2.connect(DATABASE_URI) as conn:
        with conn.cursor() as cursor:
            for row_id, embedding in data:
                cursor.execute(
                    "UPDATE articles SET embedding = %s WHERE id = %s",
                    (embedding, row_id)
                )
        conn.commit()

def main():
    # Fetch text data from the database
    text_data = fetch_text_data()
    
    # Generate embeddings and prepare data for database update
    data_for_update = [(row_id, generate_embeddings(text1, text2)) for row_id, text1, text2 in text_data]
    
    # Update the database with embeddings
    update_database_with_embeddings(data_for_update)
    
    print("Database updated with embeddings.")

if __name__ == "__main__":
    main()

