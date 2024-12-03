import streamlit as st
import tiktoken
from loguru import logger
from datetime import datetime
import requests
import json
import os


def main():
    def save_question(query):
        # 고정된 파일명 "prompt1.txt"에 덧붙이기
        prompt_file_path = os.path.join("C:\\Prompt", "prompt1.txt")

        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(prompt_file_path), exist_ok=True)

        # 현재 시간으로 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 질문을 고정된 파일에 타임스탬프와 함께 덧붙이기
        with open(prompt_file_path, 'a', encoding='utf-8') as pf:
            pf.write(f"Timestamp: {timestamp} | Question: {query}\n")
    
        #--------------------
    def save_response(result):
    # 고정된 파일명 "result1.txt"에 덧붙이기
        result_file_path = os.path.join( "C:\\Result", "result1.txt")

    # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

    # 현재 시간으로 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 응답을 고정된 파일에 타임스탬프와 함께 덧붙이기
        with open(result_file_path, 'a', encoding='utf-8') as rf:
            rf.write(f"Timestamp: {timestamp} | Response: {result}\n\n")  
#--------------------


    st.set_page_config(
        page_title='NewsBot',
        page_icon=':books:' 
        )

    st.title('_News ChatBot :red[NewsBot]_ :books:')

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 오늘의 뉴스를 물어보세요 :)"}]


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat logic
    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state['messages'].append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)
            # 질문을 기록하는 함수


        # 요청 URL
        url = "http://127.0.0.1:8000/query"
        #url = "https://secure-mayfly-instantly.ngrok-free.app/query"

        # 요청에 보낼 데이터
        data = {
            'query': query,
            'history': st.session_state['messages']  # 대화 내역 추가
        }

        # POST 요청을 보내기 위한 헤더 설정
        headers = {"Content-Type": "application/json"}

        # `result` 변수 초기화 
        result = None

        # POST 요청 보내기
        response = requests.post(url, json=data, headers=headers)
        
        save_question(query)

        # 응답 출력 디버그,  Streamlit 의 화면 출력 코드에 response.json()['result'] 입력 필요
        if response.status_code == 200:
            try:
                response_json = response.json()
                result = response_json.get('result', None)  # None이 반환되지 않도록 수정
                save_response(result)
                
                print("Response:", result)  # 응답 데이터 출력
            except json.JSONDecodeError:
                print("Error: Failed to decode JSON response")
        else:
            print(f"Error: {response.status_code}, {response.text}")


        with st.chat_message("assistant"):
            st.markdown(result if result else "뉴스를 가져오지 못했습니다. 다시 시도해주세요.")

            # 세션 상태에 응답 저장
            if result:
                st.session_state['messages'].append({"role": "assistant", "content": result})
            else:
                st.session_state['messages'].append({"role": "assistant", "content": "뉴스를 가져오지 못했습니다. 다시 시도해주세요."})



def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

if __name__ == '__main__':
    main()


