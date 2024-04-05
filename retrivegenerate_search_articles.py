import psycopg2
from sentence_transformers import SentenceTransformer

# Initialize the model
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Database configuration
DATABASE_URI = "postgresql://edb_admin:xxxxxx@xxxxxx.pg.biganimal.io/blog_data"
#TABLE_NAME = "your_table_name"  # The name of your table

def convert_text_to_vector(text):
    """Converts text to a vector using the specified model."""
    return model.encode(text).tolist()

def search_similar_texts(search_vector):
    """Searches the database for texts similar to the given vector."""
    with psycopg2.connect(DATABASE_URI) as conn:
        with conn.cursor() as cursor:
            # Prepare the SQL query using parameterized placeholders
            sql_query = f"""
            SELECT id, title, body
            FROM articles
            ORDER BY embedding <-> %s::vector ASC
            LIMIT 10;
            """
            # Execute the query with the search vector as a parameter
            cursor.execute(sql_query, (search_vector,))
            results = cursor.fetchall()
    return results

def main():
    # Input search text
    search_text = input("Enter your search text: ")
    
    # Convert search text to vector
    search_vector = convert_text_to_vector(search_text)
    
    # Perform the search
    results = search_similar_texts(search_vector)
    
    # Display the results
    print("Similar texts found:")
    for result in results:
        print(f"ID: {result[0]}, Title: {result[1]}, Body: {result[2]}")

if __name__ == "__main__":
    main()

