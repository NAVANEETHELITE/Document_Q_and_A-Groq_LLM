#UI IMPORTS
import streamlit as st

#SESSION ID IMPORTS
import uuid

#RAG IMPORTS (DOCS, SPLITS, VECTOR_DB)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

#LLM MODEL AND EMBEDDING IMPORTS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

#PROMPT TEMPLATE IMPORTS
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder

#CHAT HISTORY IMPORTS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

#STUFF DOCS AND HISTORY RETRIEVER IMPORTS
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

#EXCEPTION HANDLING IMPORTS
import traceback

#EXPORTING API KEYS
import os, dotenv
dotenv.load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_API_KEY'] = os.getenv('HF_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

#SPLIT AND EMBED INPUT FILE AND RETURN THE RETRIEVER (DB)
def embed_input_and_get_retriever(pdf_file):
    try:
        docs = PyPDFLoader(pdf_file).load()
        splits = RecursiveCharacterTextSplitter().split_documents(docs)
        hf_embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') #all-mpnet-base-v2
        db = FAISS.from_documents(documents=splits,
                                embedding=hf_embeddings)
        return db.as_retriever()
    except IndexError:
        st.error('Upload PDF with extractable text!')
        st.experimental_rerun()
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        st.error(f"{e} in line no : {tb[-1].lineno}, in file : \
            {tb[-1].filename.split('chr(92)')[-1]}") #chr(92) refers to backslash (\). We can't use '\' or '\\' here in f-strings

#GET LLM
def get_llm():
    try:
        return ChatGroq(model='llama-3.1-70b-versatile',
                        temperature=0.6)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        st.error(f"{e} in line no : {tb[-1].lineno}, in file : \
            {tb[-1].filename.split(chr(92))[-1]}")

def get_rag_chain(_retriever):
    try:
        #CONTEXTUALIZE QUESTION - PROMPT TEMPLATE
        system_q_prompt = '''
        Based on the user's latest query and the provided chat history, 
        generate a single, standalone question that incorporates relevant
        context from the chat history. Ensure the question is fully 
        understandable on its own, without requiring access to the 
        prior conversation.
        '''
        context_q_prompt = ChatPromptTemplate.from_messages([
            ('system', system_q_prompt),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}')
        ])

        #CREATE HISTORY RETRIEVER
        history_retriever = create_history_aware_retriever(llm=get_llm(),
                                                        retriever=_retriever,
                                                        prompt=context_q_prompt)

        #ANSWER QUESTION - PROMPT TEMPLATE
        system_a_prompt = '''
        Answer the question accurately, using only reliable information.
        Ensure the response is precise, clear, and avoids any unsupported
        or speculative content.
        {context}
        '''
        live_prompt = ChatPromptTemplate.from_messages([
            ('system', system_a_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])

        #STUFFING QUERIES FROM USER ALONG WITH DOCS
        docs_chain = create_stuff_documents_chain(llm=get_llm(),
                                                prompt=live_prompt)

        #HISTORY RETRIEVER CHAIN WITH STUFFED DOCS
        retrieval_chain = create_retrieval_chain(retriever=history_retriever,
                                                combine_docs_chain=docs_chain)

        #COMPLETE RAG CHAIN
        convo_rag_chain = RunnableWithMessageHistory(
            runnable=retrieval_chain, 
            get_session_history=_get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )
        
        return convo_rag_chain
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        st.error(f"{e} in line no : {tb[-1].lineno}, in file :\
            {tb[-1].filename.split(chr(92))[-1]}")

def _get_session_history(session_id:str) -> BaseChatMessageHistory:
    try:
        if session_id not in st.session_state.chat_store:
            st.session_state.chat_store[session_id] = ChatMessageHistory()
        return st.session_state.chat_store[session_id]
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        st.error(f"{e} in line no : {tb[-1].lineno}, in file :\
            {tb[-1].filename.split(chr(92))[-1]}")

def show_file_uploder():
    uploaded_file = st.sidebar.file_uploader(label='Choose a file', 
                                             type='pdf')
    if uploaded_file:
        st.session_state['uploaded_file'] = uploaded_file
        temp_pdf = r'./uploaded.pdf'
        with open(temp_pdf, 'wb') as f:
            f.write(uploaded_file.getvalue())
        return 1
    else:
        st.warning('Upload file to initiate conversation.')

def main():
    try:
        st.markdown("<h1 style='text-align: center; color: white;'>DOCUMENT Q&A</h1>",
                    unsafe_allow_html=True)
        if show_file_uploder() == 1:
            init_retriever = embed_input_and_get_retriever(pdf_file=r'./uploaded.pdf')
            if init_retriever:
                st.sidebar.success('PDF processed!')

                st.session_state.chat_store = dict()

                user_text = st.chat_input(placeholder='Message here')
                if user_text:
                    st.write('You:')
                    st.warning(user_text)
                    sess_id = str(uuid.uuid4())
                    # chat_history = _get_session_history(session_id=sess_id)
                    response = get_rag_chain(init_retriever).invoke(
                        {"input": user_text},
                        config={
                            'configurable' : {
                                'session_id' : sess_id
                            }
                        }
                    )
                    st.write('Assistant:')
                    st.success(response['answer'])
            # st.write(chat_history['messages'])
    except Exception as e:
        st.write(e)
        tb = traceback.extract_tb(e.__traceback__)
        st.error(f"{e} in line no : {tb[-1].lineno}, in file :\
            {tb[-1].filename.split(chr(92))[-1]}")

if __name__ == '__main__':
    main()
