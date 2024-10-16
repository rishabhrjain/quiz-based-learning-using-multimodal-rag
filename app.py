import streamlit as st
from utils import initialize_settings, extract_info_from_pdf, interact_with_llm, parse_llm_response

# PDF_PATH = "/Users/rishabhjain/Documents/personal/Documents/Academics/portfolio/cse_575_portfolio_report.pdf"
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
            
            st.success(f"Correct Answer: {qa_pair['answer']}")

def update_answer(key):
    st.session_state[key] = st.session_state[f"input_{key}"]


def main():
    initialize_settings()

    st.set_page_config(page_title="Quiz-based Learning System", page_icon="📚", layout="wide")
    st.title("📚 Quiz-based Learning System")
    st.header("Upload PDF and Select Difficulty")

    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("Process Files"):

        st.session_state['difficulty'] = st.select_slider(
        "Select difficulty level",
        options=["Easy", "Medium", "Difficult"],
        value="Medium"
        )
        
        st.success("File successfully uploaded!")
        
        # Extract information from PDF
        with st.spinner("Extracting information from files ..."):
            # extracted_info = extract_info_from_pdf(uploaded_file, IMAGE_DIR)
            st.session_state['extracted_info'] = extract_info_from_pdf(uploaded_files, IMAGE_DIR)
        
        st.subheader("Extracted Information")
        # st.text_area("PDF Content Preview", extracted_info[:500] + "...", height=150)
        
    # Interact with LLM
    if st.button("Generate Quiz"):
        with st.spinner("Generating quiz based on content and difficulty..."):
            # commenting the actual API call to save credits
            llm_response = interact_with_llm(st.session_state['extracted_info'], st.session_state['difficulty'])

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

if __name__ == "__main__":
    main()

# def display_quiz(quiz_data):
#     # quiz_data = json.loads(quiz_json)
    
    
#     for qa_pair in quiz_data:
#         with st.expander(f"Question {qa_pair['id']}"):
#             st.write(qa_pair['question'])
#             st.text_input("Your answer:", key=f"q{qa_pair['id']}")
#             # if st.button("Show Answer", key=f"show_answer_{qa_pair['id']}"):
#             st.success(f"Correct Answer: {qa_pair['answer']}")

# st.sidebar.header("About")
# st.sidebar.info(
#     "This app generates quizzes based on uploaded PDF content. "
#     "Select the difficulty level and upload a PDF to get started!"
# )

# st.sidebar.header("Instructions")
# st.sidebar.markdown(
#     """
#     1. Upload a PDF file
#     2. Select the difficulty level
#     3. Wait for the system to generate a quiz
#     4. Review the generated quiz in JSON format
#     """
# )
