import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from datetime import datetime
import requests

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    st.write("API 키가 설정되었습니다.")
else:
    st.warning("API 키를 입력해주세요.")
    
# 모델 초기화
if openai_api_key:  # API 키가 설정된 경우에만 모델을 초기화
    model = ChatOpenAI(model="gpt-4o-mini")
else:
    st.stop()  # API 키가 없으면 앱 실행 중단
# 3. 네이버 뉴스 API를 이용한 데이터 로드
NAVER_NEWS_API_URL = "https://openapi.naver.com/v1/search/news.json"
headers = {
    "X-Naver-Client-Id": "8Jweb61zU8SEsEq2XxMM",
    "X-Naver-Client-Secret": "ThHHHbr1Je"
}
def get_news(query, display=10):
    params = {
        "query": query,
        "display": display,
        "sort": "sim"  # 유사도 기준 정렬
    }
    response = requests.get(NAVER_NEWS_API_URL, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"뉴스 API 호출 오류: {response.status_code}")
        return None
# 4. 뉴스 데이터를 문서로 변환
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
# 5. 텍스트 청크 생성
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
def split_documents(documents):
    return text_splitter.split_documents(documents)
# 6. 임베딩 및 벡터 스토어 생성
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
def create_vector_store(splits):
    return FAISS.from_documents(documents=splits, embedding=embeddings)
# 7. 대화 관리 및 RAG 체인
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
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
chat_history = ChatMessageHistory()
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


# 8. 메인 대화 루프
def main():
    result_dir = "C:\\Users\\dlswj\\Documents\\python\\quest"  # 결과 저장 디렉토리
    news_data = get_news("오늘의 뉴스")
    if not news_data:
        return
    loader = NaverNewsLoader(news_data)
    documents = loader.load()
    splits = split_documents(documents)
    vectorstore = create_vector_store(splits)
    retriever = NewsRetriever(vectorstore)
    while True:
        query = st.text_input("질문을 입력하세요:")
        if query.strip().lower() in ["종료", "끝"]:
            st.write("대화를 종료합니다.")
            break
        # 검색 및 답변 생성
        response_docs = retriever.retrieve(query)
        prompt = ContextToPrompt(prompt_template).invoke({
            "context": response_docs,
            "question": query
        })
        response = model(prompt)
        # 대화 기록 관리
        chat_history.add_user_message(query)
        chat_history.add_ai_message(response.content)
        # 출력
        st.write(f"\n[질문] {query}")
        st.write(f"[답변] {response.content}\n")
        st.write(f"관련 뉴스 링크: {[doc.metadata['link'] for doc in response_docs]}")
        # 결과 저장
        save_results(query, response.content, result_dir)
if __name__ == "__main__":
    main()