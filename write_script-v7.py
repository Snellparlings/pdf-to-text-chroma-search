import os
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.console import Console
import torch
from transformers import AutoTokenizer, AutoModel

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2900, chunk_overlap=160)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
# embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
# Ensure embeddings are on the GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# console.print(f"Using device: {device}", style="bold blue")
# Use DataParallel to utilize multiple GPUs

# Load the model and tokenizer from Hugging Face Transformers
model_name = "BAAI/bge-m3"
# model_name="mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

if torch.cuda.device_count() > 1:
    console.print(f"Using {torch.cuda.device_count()} GPUs", style="bold blue")
    model = torch.nn.DataParallel(model)
model = model.to('cuda')

# Function to compute embeddings
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./dba2")

# Create or get the collection
collection_name = "my_collection"

if collection_name in client.list_collections():
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name)

# Process each PDF in the ./input directory
input_directory = './input'
batch_size = 60  # Adjust based on your GPU memory
for filename in os.listdir(input_directory):
    if filename.endswith('.pdf'):
        file_path = os.path.join(input_directory, filename)
        console.print(f"Processing file: {file_path}", style="bold green")
        # Convert PDF to text
        text = pdf_to_text(file_path)

        # Split text into chunks
        chunks = text_splitter.split_text(text)

        # Prepare lists for documents and IDs
        documents_list = []
        ids_list = []

        for i, chunk in enumerate(chunks):
            documents_list.append(chunk)
            ids_list.append(f"{filename}_{i}")

        # Process in batches
        for i in range(0, len(documents_list), batch_size):
            batch_docs = documents_list[i:i + batch_size]
            batch_ids = ids_list[i:i + batch_size]

            # Embed the batch
            batch_embeddings = [embeddings.embed_query(doc) for doc in batch_docs]

            # Add the data to the ChromaDB collection
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_docs,
                ids=batch_ids
            )
        console.print(f"Processed {filename}", style="bold green")

console.print("Processing complete.", style="bold blue")
