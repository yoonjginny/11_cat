import streamlit as st
import tiktoken
from loguru import logger
from langchain.memory import StreamlitChatMessageHistory
from langchain.callbacks import get_openai_callback
from datetime import datetime

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.manager import get_openai_callback


import requests
import json

def main():
    st.set_page_config(
        page_title='NewsBot',
        page_icon=':books:' 
        )

    st.title('_News ChatBot :red[NewsBot]_ :books:')

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 최신 뉴스를 물어보세요 :)"}]
    if 'chat_history' not in st.session_state: 
        st.session_state['chat_history'] = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)


        # 요청 URL
        url = "http://127.0.0.1:8000/query"
        #url = "https://secure-mayfly-instantly.ngrok-free.app/query"

        # 요청에 보낼 데이터
        data = {'query':query}

        # POST 요청을 보내기 위한 헤더 설정
        headers = {"Content-Type": "application/json"}

        # `result` 변수 초기화 
        result = None

        # POST 요청 보내기
        response = requests.post(url, json=data, headers=headers)

        # 응답 출력 디버그,  Streamlit 의 화면 출력 코드에 response.json()['result'] 입력 필요
        if response.status_code == 200:
            result=response.json()['result',None]
            print("Response:",result)
            #print("Response:", response.json()) # 원본
        else:
            print(f"Error: {response.status_code}, {response.text}")



        with st.chat_message("assistant"):
            st.session_state.chat_history.append({"role": "user", "content": query}) 
            st.session_state.chat_history.append({"role": "assistant", "content": result}) 
            st.markdown(result if result else "뉴스를 가져오지 못했습니다. 다시 시도해주세요.")

            # response가 None인 경우 대비
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "뉴스를 가져오지 못했습니다. 다시 시도해주세요."})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

if __name__ == '__main__':
    main()


