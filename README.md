# PDF to Text Chroma Search
Python scripts that converts PDF files to text, splits them into chunks, and stores their vector representations using GPT4All embeddings in a Chroma DB. It also provides a script to query the Chroma DB for similarity search based on user input.

## Requirements

- Python 3.x
- PyPDF2
- chromadb
- langchain

## Installation

1. Clone the repository:
```
git clone https://github.com/your-username/pdf-to-text-chroma-search.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Script 1: Convert PDFs to text, split into chunks, and store in Chroma DB

1. Place your PDF files in the `input` directory.
2. Run the following command to convert the PDFs to text, split them into chunks, and store their vector representations in the Chroma DB:
```
python write_script.py
# or updated write script with embeddings model running on the gpu and output returned in markdown via 'console()'
# python write_script-v7.py
```

### Script 2: Load Chroma DB and query user input

1. Run the following command to load the Chroma DB and query user input:
```
python read_script.py
# or updated read script with compatibele embeddings model from updated write script. output returned in markdown via 'console()'
# python read_script2.py
```
2. Enter your query when prompted.


### ToDo

- *1. * design a more error prone approach when PDF files lack selected text function.

- *2. * User input requested if the local db file can be overwritten in case it already exists.

- ...
