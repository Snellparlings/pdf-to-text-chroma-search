import chromadb
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.markdown import Markdown
from rich.console import Console

# Initialize console for rich output
console = Console()

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./db2")

# Retrieve the collection named 'my_collection'
collection = client.get_collection(name="my_collectionfull2")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Get user input
query = input("Enter your query: ")

# Convert the query to vector representation
query_vector = embeddings.embed_query(query)

# Query ChromaDB with the vector representation
results = collection.query(query_embeddings=query_vector, n_results=8, include=["documents"])

# Print results in markdown format
for result in results["documents"]:
    for document in result:
        md = Markdown(document)
        console.print(md)
