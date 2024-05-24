import os
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from rich.console import Console
from rich.markdown import Markdown

# Initialize console for rich output
console = Console()

# Function to convert PDF to text
def pdf_to_text(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        console.print(f"Error reading {file_path}: {e}", style="bold red")
    return text

# Initialize text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./db")

# Create or get the collection
collection_name = "my_collection"
if collection_name in client.list_collections():
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name)

# Process each PDF in the ./input directory
input_directory = './input'
for filename in os.listdir(input_directory):
    if filename.endswith('.pdf'):
        file_path = os.path.join(input_directory, filename)
        # Convert PDF to text
        text = pdf_to_text(file_path)

        # Split text into chunks
        chunks = text_splitter.split_text(text)

        # Prepare lists for documents, embeddings, and IDs
        documents_list = []
        embeddings_list = []
        ids_list = []
        
        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk)
            documents_list.append(chunk)
            embeddings_list.append(vector)
            ids_list.append(f"{filename}_{i}")
        
        # Add the data to the ChromaDB collection
        collection.add(
            embeddings=embeddings_list,
            documents=documents_list,
            ids=ids_list
        )
        console.print(f"Processed {filename}", style="bold green")

console.print("Processing complete.", style="bold blue")
