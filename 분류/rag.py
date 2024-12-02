from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import requests

from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS

from langchain.vectorstores.base import VectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.chat_message_histories import ChatMessageHistory
from datetime import datetime

# 1. 환경 변수에서 API 키 가져오기
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
naver_client_id = os.getenv("NAVER_CLIENT_ID")
naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
print(openai_api_key)
print(naver_client_id)
print(naver_client_secret)
os.environ["OPENAI_API_KEY"] = openai_api_key

# 2. 모델 초기화 (model)
model = ChatOpenAI(model="gpt-4o-mini")

# 3. 네이버 뉴스 API를 이용한 데이터 로드 (get_news -> news_data)
NAVER_NEWS_API_URL = "https://openapi.naver.com/v1/search/news.json"
headers = {
    "X-Naver-Client-Id": naver_client_id,
    "X-Naver-Client-Secret": naver_client_secret
}

def get_news(query, display=10):
    params = {
        "query": query,
        "display": display
    }
    response = requests.get(NAVER_NEWS_API_URL, headers=headers, params=params)
    news_data = response.json()
    print("API 응답 데이터: ", news_data) # 응답 데이터를 출력하여 확인
    return news_data

news_data = get_news("오늘의 뉴스")

# 4. Langchain을 이용한 뉴스 데이터 처리 (loader -> docs)
class NaverNewsLoader:
    def __init__(self, news_data):
        self.news_data = news_data
    
    def load(self):
        documents = [
            Document(
                page_content=item['title'],
                metadata={
                    "link": item['link'],
                    "description": item['description'],
                    "pubDate": item['pubDate']
                }
            )
            for item in self.news_data['items']
        ]
        return documents

  # 문서 로드 (loader)
loader = NaverNewsLoader(news_data=news_data)
docs = loader.load()

# 5. chunking (splits)
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)

splits = recursive_text_splitter.split_documents(docs)

# 6. embedding
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") # 토큰화된 문서를 모델에 입력하여 임베딩 벡터를 생성하고, 이를 평균하여 전체 문서의 벡터를 생성

# 7. vector store 생성
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# 8. retriever 생성
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# 9. 프롬프트 템플릿 정의
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following context."),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])

# 10. RAG 체인 구성
 # 디버깅을 위해 만든 클래스
class SimplePassThrough:
    def invoke(self, inputs, **kwargs):
        return inputs

 # 프롬프트 클래스 (response docs -> context)
class ContextToPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def invoke(self, inputs):
        # response_docs 내용을 trim해줌 (가독성을 높여줌)
        if isinstance(inputs, list): # inputs가 list인 경우. 즉 여러개의 문서들이 검색되어 리스트로 전달된 경우
            context_text = "\n".join([doc.page_content for doc in inputs]) # \n을 구분자로 넣어서 한 문자열로 합쳐줌
        else:
            context_text = inputs # 리스트가 아닌경우는 그냥 리턴해줌

        # 프롬프트
        formatted_prompt = self.prompt_template.format_messages( # 템플릿의 변수에 삽입해줌
            context=context_text, # {context} 변수에 context_text, 즉 검색된 문서 내용을 삽입함
            question=inputs.get("question", "")
        )
        return formatted_prompt

 # Retriever 클래스 (query)
class RetrieverWrapper:
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, inputs):
        # 0단계 : query의 타입에 따른 전처리
        if isinstance(inputs, dict): # inputs가 딕셔너리 타입일경우, question 키의 값을 검색 쿼리로 사용
            query = inputs.get("question", "")
        else: # 질문이 문자열로 주어지면, 그대로 검색 쿼리로 사용
            query = inputs
        # 1단계 : query를 리트리버에 넣어주고, response_docs를 얻어모
        response_docs = self.retriever.get_relevant_documents(query) # 검색을 수행하고 검색 결과를 response_docs에 저장
        return response_docs

 # RAG 체인 설정
rag_chain_debug = {
    "context": RetrieverWrapper(retriever), # 클래스 객체를 생성해서 value로 넣어줌
    "prompt": ContextToPrompt(contextual_prompt),
    "llm": model
}

def generate_response(query_text: str):
    try:
        # rag_chain.invoke()를 사용하여 query_text를 처리합니다.
        result = rag_chain_debug.invoke(query_text)  # query_text를 사용해 처리
        return result  # 결과 반환
    except Exception as e:
        # 예외가 발생하면 처리하고, 적절한 메시지 반환
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
