# PDF Text Retrieval and Summarization

This project extracts text from a collection of PDF documents, processes it, stores it in a vector database, and retrieves relevant passages based on a user query. The retrieved passages are then summarized to provide a coherent response.

## Requirements

- Python 3.6+
- PyMuPDF
- SentenceTransformers
- FAISS
- Transformers
- numpy
- re

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/pdf-text-retrieval.git
   cd pdf-text-retrieval

## Code Workflow

1. Text Extraction
- The extract_text_from_pdf function reads the text content from each page of the PDF files. 
- It uses the PyMuPDF library (fitz) to open the PDF and iterate through each page. 
- The function can also filter out pages that contain specific keywords

2. Text Splitting
- split_text_into_sections function splits the text into smaller, manageable sections. 
- This uses regular expressions that match different numbering schemes (e.g., "1 ", "1.1 ", "a)"). 

3. Vectorization
- Each section is converted into a numerical vector representation using a pre-trained model from the SentenceTransformers library. 

4. Indexing
- The vectorized sections are stored in a FAISS (Facebook AI Similarity Search) index. 

5. Query Handling
- It vectorizes the query text and searches the FAISS index for the most similar text sections. 
- The retrieve_relevant_passages function handles this process, returning a list of relevant text sections.

6. Summarization
- The summarize_passages function uses a summarization model from the Transformers library ( "facebook/bart-large-cnn") to condense these sections into shorter summaries. 

### Potential Improvements
- Detailed text Preprocessing methods- removing stopwords, special characters
- Text splitting done more intelligently
- Fine-tune the summarization model on a domain-specific dataset.
- Better contextual embeddings