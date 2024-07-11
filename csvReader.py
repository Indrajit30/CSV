import os
import streamlit as st
from PIL import Image

## prompts
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

logo_path = "./avendus.png"
logo_url = "https://media.licdn.com/dms/image/D4D0BAQFIdC9CJFGppg/company-logo_200_200/0/1687269247270?e=2147483647&v=beta&t=83wAMv0VQSvsVbvyE7g80oKJJ6MgILP3ExSsIIGT5wg"

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
else:
    logo = logo_url

system_template = r'''
Use the following piece of context to answer the questions.

You are a highly skilled financial expert with extensive knowledge and expertise in analyzing financial data, calculating key financial metrics, and providing insights. You are tasked with assisting users in understanding their financial performance by calculating metrics such as EBITDA, YoY profits, and other relevant financial indicators. Users will provide you with basic financial information like sales, revenue, and expenses for different periods (e.g., quarters, years). Based on this information, you will perform accurate calculations.

Your tasks include, but are not limited to:

Calculating EBITDA:

EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization.
Formula: EBITDA = Operating Profit + Depreciation + Amortization
Calculating Year-over-Year (YoY) Profits:

YoY Profit measures the percentage change in profits from one period to the same period in the previous year.
Formula: YoY Profit = ((Current Period Profit - Previous Period Profit) / Previous Period Profit) * 100
Analyzing Revenue and Sales Growth:

Calculate the percentage growth in sales and revenue over different periods.
Formula for growth: Growth Rate = ((Current Period Value - Previous Period Value) / Previous Period Value) * 100
Calculating Operating Profit Margin (OPM):

OPM measures the percentage of revenue that remains after covering operating expenses.
Formula: OPM = (Operating Profit / Revenue) * 100
Net Profit Margin:

Net Profit Margin measures the percentage of revenue that remains as profit after all expenses, including taxes and interest.
Formula: Net Profit Margin = (Net Profit / Revenue) * 100
Providing insights and explanations:

Interpret the calculated metrics to provide meaningful insights.
Explain the significance of each metric and how it reflects the company’s financial health.
Offer suggestions for improvement if needed.

If something is not provided in the data then try to calculate it using the given data using the formulae given above.
---------------
Context: ```{context}```
'''
user_template = '''
Question: ```{question}```
'''
messages =[
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

qa_prompt = ChatPromptTemplate.from_messages(messages)

##functions
def load_csv(file):
    from langchain.document_loaders import CSVLoader
    loader = CSVLoader(file_path=file)
    data = loader.load()
    return data

def chunk_data(data,chunk_size=400):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = 100
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def insert_or_fetch_embeddings(index_name,chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small",dimensions = 1536)

    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings..',end="")
        vector_store = Pinecone.from_existing_index(index_name=index_name,embedding = embeddings)
        print('Ok')

    else:
        print(f'Creating index {index_name} and embeddings....',end="")
        pc.create_index(
            name = index_name,
            dimension = 1536,
            metric = "cosine",
            spec = PodSpec(
                environment = 'gcp-starter'
            )
        )
        vector_store = Pinecone.from_documents(chunks,embedding=embeddings,index_name = index_name)
        print('Ok')
        return vector_store
    
def ask_questions(q,chain):
    result = chain.invoke({'question':q})
    return result['answer']

def load_embeddings_pinecone(index_name):
    from langchain.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
    # Initialize Pinecone 
    vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    return vector_store
    
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.4)

    if 'vs' not in st.session_state:
        st.session_state['vs'] = None

    st.image(logo, width=100)
    st.header("AI Financial Analyst Co-pilot")
    with st.sidebar:
        # uploaded_file = st.file_uploader('Upload a file: ',type=['csv'])
        uploaded_file = st.text_input('Enter the path of the folder containing your files:')
        
        if uploaded_file:
            data = load_csv(uploaded_file)
            chunks = chunk_data(data)
            vector_store = insert_or_fetch_embeddings("excel",chunks)
            st.session_state['vs'] = vector_store

    q = st.text_input("Enter the question")
    submit = st.button("submit")
    if submit:
        if q:
            with st.spinner("Running..."):
                if st.session_state['vs'] is not None:
                    vector_store = st.session_state['vs']
                else:
                    vector_store = load_embeddings_pinecone("excel")
                    st.session_state['vs'] = vector_store

                llm = ChatOpenAI(model="gpt-4-turbo",temperature=0)
                retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':5})
                memory = ConversationBufferMemory(memory_key="chat_history",return_messages = True)
                crc = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff",
                    combine_docs_chain_kwargs={'prompt': qa_prompt},
                    verbose=False,
                    memory=memory
                )
                answer = ask_questions(q, crc)
            st.text_area('Answer:', value=answer, height=300)

            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-"*100}\n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label="Chat history",value=h, key='history',height = 300)