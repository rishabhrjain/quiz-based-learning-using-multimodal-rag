
import json
import fitz
import os
import re
import streamlit as st


from llama_index.core import Settings, StorageContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

# Set up the Ollama LLM
llm = Ollama(model="llama3.2:1b", temperature=0.1)

def initialize_settings():
    
    # Configure LlamaIndex to use the Ollama LLM
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    if 'quiz_data' not in st.session_state:
        st.session_state['quiz_data'] = None

def extract_info_from_pdf(pdf_paths, image_dir):
    """
    Extract infor from pdf. Currently this only extracts text. Ideally image should be read and added to text as well

    Args:
        pdf_paths (_type_): _description_
        image_dir (_type_): _description_

    Returns:
        _type_: _description_
    """

    documents = []
    for file in pdf_paths:
        doc = fitz.open(stream=file.read())
        
        # Create directory for images if it doesn't exist
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        # Initialize lists to store text and image info
        all_text = []
        image_info = []
        
        # Iterate through pages
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            all_text.append(f"Page {page_num + 1}:\n{text}\n")
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"page{page_num + 1}_img{img_index}.{image_ext}"
                image_path = os.path.join(image_dir, image_filename)
                
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                
                image_info.append(f"Page {page_num + 1}, Image {img_index}: {image_filename}")
        
        # Close the document
        doc.close()

        documents.append(all_text)

        
    return documents

def interact_with_llm(text, difficulty_level):
    # Placeholder function for LLM interaction
    # Replace this with your actual LLM implementation

    prompt = """
    Create 5 question answer pairs from context in a valid json format as given in the below context. 

    JSON format:
    {
    "id": "1",
    "question": "question 1",
    "answer": "Answer 1"
    }

    """

    ## Get only the text from first document right now
    context = f"""
    Context: 
    {text[0][0]}
    """

    prompt = prompt + context
    # print("\n\n ############################# prompt is : \n", prompt)
    llm_response = llm.complete(prompt=prompt)

    return llm_response


def parse_llm_response(llm_response):
    # Extract JSON objects from the response - return as a list of dict. Each dict has question and answer keys
    json_objects = re.findall(r'\{[^}]+\}', llm_response.text)

    # Parse each JSON object
    qa_pairs = []
    for json_str in json_objects:
        try:
            qa_pair = json.loads(json_str)
            qa_pairs.append(qa_pair)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {json_str}")

    return qa_pairs