# PDF Text Retrieval and Summarization  

## Requirements
- PyMuPDF (fitz)
- ChromaDB
- sentence_transformers
- transformers

```bash
pip install PyMuPDF chromadb sentence-transformers transformers
```
- `./chroma`: Directory where the ChromaDB is persisted.

## Functions

### `extract_text_from_pdfs(pdf_paths)`

- Extracts text from a list of PDF files. 


### `split_text_into_sections(text, model)`

- Splits the extracted text into sections based on regular expression patterns. 
- encodes these sections into vectors using a sentence transformer model (all-MiniLM-L6-v).


### `create_vector_db(sections, vectors, persist_directory)`

- Creates a vector database using ChromaDB to store document sections and their corresponding vector embeddings.


### `vectorize_query(query, model)`

- Encodes a query string into a vector using the same model used for section encoding.


### `retrieve_relevant_passages(query, collection, model, top_k=5)`

- Retrieves the most relevant passages from the database based on the query vector.


### `summarizer`

- A pre-loaded Hugging Face pipeline for summarization, (facebook/bart-large-cnn)"


### `clean_passages(passages)`

- Cleans the retrieved passages.


### `summarize_passages(cleaned_passages)`

Summarizes the cleaned passages using the pre-loaded summarization model.


### `answer_query(query, collection, model)`

- Processes a query by retrieving relevant passages, cleaning them, and summarizing the result.

To run the script:
1. Populate the `pdf_paths` list with the paths to the PDF files.
2. Execute the script. This will automatically extract text, split it into sections, create a vector database, and allow querying based on a specified example query.


### Potential Improvements
- Detailed text Preprocessing methods- removing stopwords, special characters
- Text splitting done more intelligently
- Fine-tune the summarization model on a domain-specific dataset.
- Better contextual embeddings