import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

# Add the missing get_vectorstore function
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


# Modify the get_conversation_chain function to accept API keys
def get_conversation_chain(vectorstore, openai_api_key, huggingfacehub_api_token):
    llm = ChatOpenAI(api_key=openai_api_key)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}, api_token=huggingfacehub_api_token)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def get_text_chunks(text):
    # Split the text into chunks of a fixed size (e.g., 1000 characters)
    chunk_size = 1000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to get OpenAI API Key and Hugging Face Hub API Token from user input



def get_api_keys():
    OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN = None, None

    with st.form("user_input"):
        OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key:", placeholder="sk-XXXX", type='password')
        HUGGINGFACEHUB_API_TOKEN = st.text_input("Enter your Hugging Face Hub API Token:", placeholder="your-token", type='password')
        submitted = st.form_submit_button("Submit")

    if submitted:
        if not OPENAI_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
            st.info("Please fill out both OpenAI API Key and Hugging Face Hub API Token to proceed.")
            st.stop()

        # Initialize conversation chain if it's None
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(vectorstore, OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN)

    return OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    # Get API keys from user input
    OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN = get_api_keys()

    # Check and initialize the conversation if it's None
    if st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain(vectorstore, OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN)

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Rest of your code...

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN)

# Modify get_conversation_chain function to accept API keys
def get_conversation_chain(vectorstore, openai_api_key, huggingfacehub_api_token):
    llm = ChatOpenAI(api_key=openai_api_key)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}, api_token=huggingfacehub_api_token)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

if __name__ == '__main__':
    main()
