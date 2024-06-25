import os
import fitz  # PyMuPDF
import re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline

persist_directory = "./chroma"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF documents
def extract_text_from_pdfs(pdf_paths,keywords_to_ignore=["Inhalt"]):
    text=""
    for pdf in pdf_paths:
        document = fitz.open(pdf)
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            page_text = page.get_text()
        
            if any(keyword in page_text for keyword in keywords_to_ignore):
                continue
            text += page_text
    return text

# Split text into sections
def split_text_into_sections(text,model):
    patterns = [
        r'^\d+\s+',           
        r'^\d+\.\d+\s+',     
        r'^[a-zA-Z]+\)+\s+',     
    ]
    combined_pattern = '|'.join(patterns)
    regex_pattern = re.compile(combined_pattern, re.MULTILINE)
    # Split the text based on the pattern
    sections = []
    for section in re.split(regex_pattern, text):
        if section.strip():
            sections.append(section.strip())
    vectors = model.encode(sections)  
    return sections, vectors

# Step 4: Create and populate ChromaDB
def create_vector_db(sections, vectors, persist_directory):
    chroma_db = chromadb.PersistentClient(path=persist_directory, settings=Settings(allow_reset=True))
    collection = chroma_db.get_or_create_collection("documents")
    collection.add(
        documents=sections,
        ids=[str(i) for i in range(len(sections))],
        embeddings=vectors.tolist()  
    )
    return collection

# Vectorize the query
def vectorize_query(query, model):
    query_vector = model.encode([query])
    return query_vector

# Retrieve relevant passages
def retrieve_relevant_passages(query, collection, model, top_k=5):
    query_vector = vectorize_query(query, model)
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=top_k,
    )
    results_list = results['documents'][0]
    passages = ""
    for item in results_list:
        passages += item
    return passages

# summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# cleaning of passages
def clean_passages(passages):
    cleaned_passages = re.sub(r'\.{2,}', ' ', passages)
    cleaned_passages = re.sub(r'\s*\n\s*', ' ', cleaned_passages)
    cleaned_passages = re.sub(r'\s+', '', cleaned_passages)  
    cleaned_passages = re.sub(r'\s+', ' ', cleaned_passages).strip()
    return cleaned_passages

# summarize relevant passages
def summarize_passages(cleaned_passages):
    try:
        summarized_response = summarizer(cleaned_passages, max_length=800, min_length=150, do_sample=False)
        return summarized_response[0]['summary_text']
    except Exception as e:
        print("Error during summarization:", e)
        return cleaned_passages  

# 
def answer_query(query, collection, model):
    passages = retrieve_relevant_passages(query, collection, model)
    response = "".join(passages)
    cleaned_passages = clean_passages(response)
    summarized_response = summarize_passages(cleaned_passages)
    return summarized_response

if __name__ == "__main__":
    pdf_paths = ["documents/Basispaket+WeitBlick.pdf","documents/pa_d_1006_iii_11_211.pdf"]

    documents = extract_text_from_pdfs(pdf_paths)
    sections, vectors = split_text_into_sections(documents, model)
    collection = create_vector_db(sections, vectors, persist_directory)

    query = input("Please enter your query: ")
    response = answer_query(query, collection, model)
    print("................................Summarized Response:............................", response)
