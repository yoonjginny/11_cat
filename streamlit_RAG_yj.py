import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from datetime import datetime
import requests
from loguru import logger
from langchain.memory import StreamlitChatMessageHistory
from langchain.callbacks import get_openai_callback

# Streamlit 페이지 설정
st.set_page_config(
    page_title='NewsBot',
    page_icon=':books:'
)

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
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

# 뉴스 데이터를 문서로 변환
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

# 텍스트 청크 생성
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

def split_documents(documents):
    return text_splitter.split_documents(documents)

# 임베딩 및 벡터 스토어 생성
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def create_vector_store(splits):
    return FAISS.from_documents(documents=splits, embedding=embeddings)

# 대화 관리 및 RAG 체인 설정
class ContextToPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def invoke(self, inputs):
        if isinstance(inputs, list):
            context_text = "\n".join([f"{doc.page_content}\n{doc.metadata['link']}" for doc in inputs])
        else:
            context_text = inputs
        return self.prompt_template.format_messages(context=context_text, question=inputs.get("question", ""))

class NewsRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query):
        return self.vectorstore.as_retriever().get_relevant_documents(query)

prompt_template = ChatPromptTemplate.from_template(
    "다음 정보를 바탕으로 질문에 대한 답변을 작성하세요.\n\n"
    "뉴스 기사:\n{context}\n\n"
    "질문: {question}\n"
    "답변:"
)

#chat_history = StreamlitChatMessageHistory()

# 결과 저장 함수
def save_results(question, response, result_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_file_path = os.path.join(result_dir, "Prompt", "prompt1.txt")
    os.makedirs(os.path.dirname(prompt_file_path), exist_ok=True)
    with open(prompt_file_path, 'a', encoding='utf-8') as pf:
        pf.write(f"Question: {question}\n")
    result_file_path = os.path.join(result_dir, "result", f"result_{timestamp}.txt")
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    with open(result_file_path, 'a', encoding='utf-8') as rf:
        rf.write(f"Response: {response}\n\n")

# Streamlit 메인 함수
def main():
    st.title('_News ChatBot :red[NewsBot]_ :books:')

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 최신 뉴스를 물어봐주세요 :)"}]

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
            if news_data:
                news_loader = NaverNewsLoader(news_data)
                docs = news_loader.load()
                splits = split_documents(docs)
                vectorstore = create_vector_store(splits)
                retriever = NewsRetriever(vectorstore)
                relevant_docs = retriever.retrieve(query)

                context_to_prompt = ContextToPrompt(prompt_template)
                context = context_to_prompt.invoke(relevant_docs)
                response = model(context)

                with get_openai_callback() as cb:
                    st.session_state.chat_history.add_message("user", query)
                    st.session_state.chat_history.add_message("assistant", response.content)
                    st.markdown(response.content)

                save_results(query, response.content, "results")

            # response가 None인 경우 대비
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "뉴스를 가져오지 못했습니다. 다시 시도해주세요."})

if __name__ == '__main__':
    main()
