import streamlit as st
import tiktoken
from loguru import logger
from langchain.memory import StreamlitChatMessageHistory
from langchain.callbacks import get_openai_callback
from datetime import datetime

# 필요한 라이브러리와 모듈 임포트
from dotenv import load_dotenv
import os
import requests

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 환경 변수에서 API 키 가져오기
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
naver_client_id = os.getenv("NAVER_CLIENT_ID")
naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")

os.environ["OPENAI_API_KEY"] = openai_api_key

# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini")

# 네이버 뉴스 API를 이용한 데이터 로드
NAVER_NEWS_API_URL = "https://openapi.naver.com/v1/search/news.json"
headers = {
    "X-Naver-Client-Id": naver_client_id,
    "X-Naver-Client-Secret": naver_client_secret
}

def get_news(query, display=10):
    params = {
        "query": query,
        "display": display,
        "sort": "sim"
    }
    response = requests.get(NAVER_NEWS_API_URL, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"뉴스 API 호출 오류: {response.status_code}")
        return None

def main():
    st.set_page_config(
        page_title='NewsBot',
        page_icon=':robot:'
    )

    st.title('_News ChatBot :red[NewsBot]_ :robot:')

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 최신 뉴스를 물어봐주세요 :)"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            news_data = get_news(query)
            response = None  # response 변수를 초기화
            if news_data and 'items' in news_data:
                class NaverNewsLoader:
                    def __init__(self, news_data):
                        self.news_data = news_data
                    
                    def load(self):
                        documents = [
                            Document(
                                page_content=f"{item['title']}\n{item['description']}",
                                metadata={"link": item['link'], "pubDate": item['pubDate']}
                            )
                            for item in self.news_data.get('items', [])
                        ]
                        return documents

                # 문서 로드
                loader = NaverNewsLoader(news_data=news_data)
                docs = loader.load()

                # 텍스트 청크 생성
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
                splits = text_splitter.split_documents(docs)

                # 임베딩 및 벡터 스토어 생성
                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

                # 리트리버 생성
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

                # 프롬프트 템플릿 정의
                prompt_template = ChatPromptTemplate.from_template(
                    "다음 정보를 바탕으로 질문에 대한 답변을 작성하세요.\n\n"
                    "뉴스 기사:\n{context}\n\n"
                    "질문: {question}\n"
                    "답변:"
                )

                # 프롬프트 클래스 정의
                class ContextToPrompt:
                    def __init__(self, prompt_template):
                        self.prompt_template = prompt_template

                    def invoke(self, inputs):
                        context_text = "\n".join([f"{doc.page_content}\n{doc.metadata['link']}" for doc in inputs])
                        formatted_prompt = self.prompt_template.format_messages(
                            context=context_text,
                            question=inputs.get("question", "")
                        )
                        return formatted_prompt

                # 리트리버 클래스 정의
                class RetrieverWrapper:
                    def __init__(self, retriever):
                        self.retriever = retriever

                    def invoke(self, inputs):
                        query = inputs.get("question", "")
                        response_docs = self.retriever.get_relevant_documents(query)
                        return response_docs

                # RAG 체인 설정
                rag_chain_debug = {
                    "context": RetrieverWrapper(retriever),
                    "prompt": ContextToPrompt(prompt_template),
                    "llm": model
                }

                # 리트리버로 question에 대한 검색 결과를 response_docs에 저장
                response_docs = rag_chain_debug["context"].invoke({"question": query})

                # 프롬프트에 질문과 response_docs를 넣어줌
                prompt_messages = rag_chain_debug["prompt"].invoke({
                    "context": response_docs,
                    "question": query
                })

                # 완성된 프롬프트를 LLM에 넣어줌
                response = rag_chain_debug["llm"].invoke(prompt_messages)

                with get_openai_callback() as cb:
                    st.session_state.chat_history.add_message("user", query)
                    st.session_state.chat_history.add_message("assistant", response.content)
                    st.markdown(response.content)

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
