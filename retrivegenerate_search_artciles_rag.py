from sentence_transformers import SentenceTransformer
import psycopg2
from transformers import pipeline
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter('ignore', InsecureRequestWarning)

# Configuration
DATABASE_URI = "postgresql://edb_admin:xxxxx@xxxxxx.pg.biganimal.io/blog_data"
VECTOR_DIMENSION = 384  # Ensure this matches your vector dimensions

# Initialize the Sentence Transformer model for encoding queries
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the text generation model
generator = pipeline('text-generation', model='gpt2')

def retrieve_documents(query, top_k=5):
    vector = model.encode([query])[0].tolist()  # Convert query to vector and then to list

 # Convert the NumPy array to a list of Python floats
    #vector_list = vector.tolist()

    conn = psycopg2.connect(DATABASE_URI)
    cursor = conn.cursor()

    # Adjust the SQL query to match your database schema and ensure proper vector comparison
    cursor.execute("SELECT title,body FROM articles ORDER BY embedding <-> %s::vector LIMIT %s", (vector, top_k))

    results = cursor.fetchall()
    cursor.close()
    conn.close()

    # Return only the document contents
    return [result[0] for result in results]

def generate_answer(documents):
    # Combine the documents into a single text context
    context = " ".join(documents)
    # Generate an answer based on the combined context
    answers = generator(context, num_return_sequences=1, truncation=True, max_new_tokens=50)
    return answers[0]['generated_text']

if __name__ == "__main__":
    query = input("Enter your query: ")
    documents = retrieve_documents(query)
    if documents:
        answer = generate_answer(documents)
        print("Generated Answer:", answer)
    else:
        print("No relevant documents found.")

