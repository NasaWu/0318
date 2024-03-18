

#####在大爷的推荐下实现 用户prompt->用户选择question type->从type中RAG->answer
import streamlit as st
import pandas as pd
import numpy as np
import json
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from modelscope import snapshot_download
from langchain_community.vectorstores import FAISS
import pickle  #to load a saved modelimport base64  #to open .gif files in streamlit app
from functions import retrieve_top_docs
from functions import extract_indices
from functions import get_selected_rows

########
loader = CSVLoader(file_path='D:/Programe/AIGC/0312/test0312question.csv', encoding='gbk')
documents = loader.load()
df1 = pd.read_csv('D:/Programe/AIGC/0312/test0312_1.csv', encoding='gbk')
df2 = pd.read_csv('D:/Programe/AIGC/0312/test0312_2.csv', encoding='gbk')
df3 = pd.read_csv('D:/Programe/AIGC/0312/test0312_3.csv', encoding='gbk')
df4 = pd.read_csv('D:/Programe/AIGC/0312/test0312_4.csv', encoding='gbk')
df5 = pd.read_csv('D:/Programe/AIGC/0312/test0312_5.csv', encoding='gbk')
df6 = pd.read_csv('D:/Programe/AIGC/0312/test0312_6.csv', encoding='gbk')
df7 = pd.read_csv('D:/Programe/AIGC/0312/test0312_7.csv', encoding='gbk')
df8 = pd.read_csv('D:/Programe/AIGC/0312/test0312_8.csv', encoding='gbk')
df9 = pd.read_csv('D:/Programe/AIGC/0312/test0312_9.csv', encoding='gbk')
df10 = pd.read_csv('D:/Programe/AIGC/0312/test0312_10.csv', encoding='gbk')
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
model_dir = snapshot_download("AI-ModelScope/bge-large-zh-v1.5", revision='master')
embedding_path=model_dir
embeddings = HuggingFaceBgeEmbeddings(model_name = embedding_path)

vectorstore = FAISS.from_documents(
    docs,
    embedding= embeddings
)
retriever = vectorstore.as_retriever()

########
@st.cache(suppress_st_warning=True)
def get_fvalue(val):    
	feature_dict = {"No":1,"Yes":2}    
	for key,value in feature_dict.items():        
		if val == key:            
			return value
def get_value(val,my_dict):    
	for key,value in my_dict.items():        
		if val == key:            
			return value


############
st.title('IPS AMP 操作问答:') 
    # 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []
# 显示已有的聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# 聊天输入框
if prompt := st.chat_input("What is up?"):
    top_docs = retrieve_top_docs(prompt)
    indices= extract_indices(top_docs)
    
    # 将用户的输入添加到聊天记录中
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    direction = st.radio(
        "请选择您想要询问的问题方向：",
        ('IPS AMP常见问题', '发票问题', 'MKT Brand Team伙伴如何申请IPS账号',
        'MKT brand Team伙伴如何登录IPS','创建/修改采购订单','审批采购订单',' 确认采购订单',
        '维护收货单','审批收货单','系统自动同步订单及收货至AMP')
                        )
    if direction=="IPS AMP常见问题":
        answer = get_selected_rows(df1,indices)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
    elif direction=="发票问题":
        answer = get_selected_rows(df1,indices)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
    elif direction=="MKT Brand Team伙伴如何申请IPS账号":
        answer = get_selected_rows(df1,indices)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
    elif direction=="MKT brand Team伙伴如何登录IPS":
        answer = get_selected_rows(df1,indices)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
    elif direction=="创建/修改采购订单":
        answer = get_selected_rows(df1,indices)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
    elif direction=="审批采购订单":
        answer = get_selected_rows(df1,indices)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
    elif direction=="确认采购订单":
        answer = get_selected_rows(df1,indices)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
    elif direction=="维护收货单":
        answer = get_selected_rows(df1,indices)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
    elif direction=="审批收货单":
        answer = get_selected_rows(df1,indices)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
    else:
        answer = get_selected_rows(df1,indices)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])
    