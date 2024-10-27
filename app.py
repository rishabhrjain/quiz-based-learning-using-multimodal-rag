import streamlit as st
from utils import initialize_settings, extract_info_from_pdf, interact_with_llm, parse_llm_response, create_weaviate_index
from llama_index.core import Settings

IMAGE_DIR = "extracted_images"

########### STREAMLIT section #######

# You might need to install additional libraries for LLM interaction
def display_quiz(quiz_data):
    if quiz_data is None:
        return None
    
    for qa_pair in quiz_data:
        with st.expander(f"Question {qa_pair['id']}"):
            st.write(qa_pair['question'])
            
            # Use session state to store user inputs
            if f"q{qa_pair['id']}" not in st.session_state:
                st.session_state[f"q{qa_pair['id']}"] = ""
            
            user_answer = st.text_input(
                "Your answer:", 
                key=f"input_q{qa_pair['id']}",
                value=st.session_state[f"q{qa_pair['id']}"],
                on_change=update_answer,
                args=(f"q{qa_pair['id']}",)
            )
            if st.button("Show answer", key=qa_pair['id']):
                st.success(f"Correct Answer: {qa_pair['answer']}")

def update_answer(key):
    st.session_state[key] = st.session_state[f"input_{key}"]

def update_temperature():
    st.session_state.temp_value = st.session_state.temperature


def main():
    initialize_settings()

    st.set_page_config(page_title="Quiz-based Learning System", page_icon="📚", layout="wide")
    st.title("📚 Quiz-based Learning System")
    col1, col2 = st.columns([1, 2])

    # Quiz section
    with col1:
        st.header("Upload PDF and Select Difficulty")

            # Initialize session state variables if they don't exist
        if 'difficulty' not in st.session_state:
            st.session_state.difficulty = "Medium"
        if 'temperature' not in st.session_state:
            st.session_state.temperature = 0.2


        st.session_state['difficulty'] = st.select_slider(
            "Select difficulty level",
            options=["Easy", "Medium", "Difficult"],
            value="Medium"
            )
        
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            key="temperature",
            on_change=update_temperature
        )
            
        Settings.llm.temperature = st.session_state.temperature

        uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

        if uploaded_files and st.button("Process Files"):
            
            st.success("File successfully uploaded!")
            
            # Extract information from PDF
            with st.spinner("Extracting information from files ..."):
                # extracted_info = extract_info_from_pdf(uploaded_file, IMAGE_DIR)
                st.session_state['extracted_info'] = extract_info_from_pdf(uploaded_files, IMAGE_DIR)

                # create Vector database index
                # st.session_state['index'] = create_index(st.session_state['extracted_info'])
                st.session_state['index'] = create_weaviate_index(st.session_state['extracted_info'])
            
            st.subheader("Extracted Information")
            # st.text_area("PDF Content Preview", extracted_info[:500] + "...", height=150)
            
        # Interact with LLM
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz based on content and difficulty..."):
                # commenting the actual API call to save credits
                # print("\n ##### Extracted info: \n\n", st.session_state['extracted_info'])
                llm_response = interact_with_llm(st.session_state['extracted_info'][0], st.session_state['difficulty'])

                # Parse LLM response
                parsed_response = parse_llm_response(llm_response)
                
                st.session_state['quiz_data'] = parsed_response
            
                st.subheader("Generated Quiz")
                display_quiz(st.session_state['quiz_data'])
                
        elif 'quiz_data' in st.session_state:
            st.subheader("Generated Quiz")
            display_quiz(st.session_state['quiz_data'])
        
        # Add a clear button
        if st.button("Clear Chat"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Query-Answer using RAG
    with col2:

        if 'index' in st.session_state:
            st.title("Chat")
            if 'history' not in st.session_state:
                st.session_state['history'] = []
            
            # llm and mebedding model is taken from initialize_settings()
            query_engine = st.session_state['index'].as_query_engine(similarity_top_k=10, streaming=True, 
                                                                     embed_model = Settings.embed_model)

            user_input = st.chat_input("Enter your query:")

            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state['history']:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state['history'].append({"role": "user", "content": user_input})
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    response = query_engine.query(user_input)
                    for token in response.response_gen:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                st.session_state['history'].append({"role": "assistant", "content": full_response})

            # Add a clear button
            if st.button("Clear Queries"):
                st.session_state['history'] = []
                st.rerun()


        # Add a clear button
        # if st.button("Clear Chat"):
        #     for key in list(st.session_state.keys()):
        #         del st.session_state[key]
        #     st.rerun()

if __name__ == "__main__":
    main()

