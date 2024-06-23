import fitz  
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import re


# Extract text from PDFs , ignore certain words
def extract_text_from_pdf(file_path,keywords_to_ignore=["Inhalt"]):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        page_text = page.get_text()
        
        if any(keyword in page_text for keyword in keywords_to_ignore):
            continue
        
        text += page_text
    return text

# Split exttracted text into sections, based on regex patterns
def split_text_into_sections(text):
    
    patterns = [
        r'^\d+\s+',           
        r'^\d+\.\d+\s+',     
        r'^[a-zA-Z]+\)',    
    ]

    # Combine into a single regex pattern
    combined_pattern = '|'.join(patterns)
    regex_pattern = re.compile(combined_pattern, re.MULTILINE)

    # Split the text based on the combined pattern
    sections = re.split(regex_pattern, text)

    # Strip whitespaces
    sections = []
    for section in re.split(regex_pattern, text):
        if section.strip():
            sections.append(section.strip())
    return sections

# Get relevant passages based on the query
def retrieve_relevant_passages(query, top_k=5):
    query_vector = model.encode([query])[0]
    distances, indices = index.search(np.array([query_vector]), top_k)
    results = [paragraph_dict[i] for i in indices[0]]

    unique_results = list(dict.fromkeys(results))
    return unique_results

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_passages(passages, max_length=500):
    summarized_passages = []
    for passage in passages:
        summary = summarizer(passage, max_length=max_length, min_length=30, do_sample=False)
        summarized_passages.append(summary[0]['summary_text'])
    return summarized_passages

def generate_response(query, top_k=5):
    relevant_passages = retrieve_relevant_passages(query, top_k)
    summarized_passages = summarize_passages(relevant_passages)
    response = " ".join(summarized_passages)
    
    cleaned_response=re.sub(r'\s+', ' ', response)
    return cleaned_response.strip()

pdf1_text = extract_text_from_pdf("documents/Basispaket+WeitBlick.pdf")
pdf2_text = extract_text_from_pdf("documents/pa_d_1006_iii_11_211.pdf")

pdf1_sections = split_text_into_sections(pdf1_text)
pdf2_sections = split_text_into_sections(pdf2_text)
all_sections = pdf1_sections + pdf2_sections

# Vectorize sections and create FAISS index
model = SentenceTransformer('all-mpnet-base-v2')
section_vectors = model.encode(all_sections)
index = faiss.IndexFlatL2(section_vectors.shape[1])
index.add(np.array(section_vectors))
paragraph_dict = {}
for i, section in enumerate(all_sections):
  paragraph_dict[i] = section

query = input("Geben Sie Ihre Anfrage ein:")
response = generate_response(query)
print("Response: ",response)
