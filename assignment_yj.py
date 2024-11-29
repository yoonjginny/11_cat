# 1. 사용 환경 준비 
import os
from getpass import getpass
os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key 입력: ")

# env로 따로 저장 


# 2. 모델 초기화 (model)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")

# 3. 네이버 뉴스 API를 이용한 데이터 로드 (get_news -> news_data)
import requests

NAVER_NEWS_API_URL = "https://openapi.naver.com/v1/search/news.json"
headers = {
    "X-Naver-Client-Id": "8Jweb61zU8SEsEq2XxMM",
    "X-Naver-Client-Secret": "ThHHHbr1Je"
}

def get_news(query, display=10):
    params = {
        "query": query,
        "display": display
    }
    response = requests.get(NAVER_NEWS_API_URL, headers=headers, params=params)
    return response.json()

news_data = get_news("오늘의 뉴스")

# 4. Langchain을 이용한 뉴스 데이터 처리 (loader -> docs)
from langchain_core.documents import Document

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
from langchain.text_splitter import RecursiveCharacterTextSplitter

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)

splits = recursive_text_splitter.split_documents(docs)

# 6. embedding
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") # 토큰화된 문서를 모델에 입력하여 임베딩 벡터를 생성하고, 이를 평균하여 전체 문서의 벡터를 생성

# 7. vector store 생성
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# 8. retriever 생성
from langchain.vectorstores.base import VectorStore

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# 9. 프롬프트 템플릿 정의
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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

# 11. 챗봇 구동
while True:
    print("========================")

    # 0. 질문을 받아서 query에 저장함
    query = input("질문을 입력하세요 : ")

    # 1. 리트리버로 question에 대한 검색 결과를 response_docs에 저장함
    response_docs = rag_chain_debug["context"].invoke({"question": query})

    # 2. 프롬프트에 질문과 response_docs를 넣어줌
    prompt_messages = rag_chain_debug["prompt"].invoke({
        "context": response_docs,
        "question": query
    })

    # 3. 완성된 프롬프트를 LLM에 넣어줌
    response = rag_chain_debug["llm"].invoke(prompt_messages)

    print("\n답변:")
    print(response.content)

# 12. 대화 내용 저장
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chat_models import ChatOpenAI

chat_history = ChatMessageHistory()

# chat_history.messages.append(SystemMessage(content='너의 이름은 햄식이이고, 아주 귀여운 햄스터야. 모든 말을 햄으로 끝내.'))


while True:
    user_input=input('사용자:')
    if user_input in ['그만','잘있어']:
     print('종료')
     break
    # 사용자 메세지를 대화 기록에 추가
    chat_history.add_user_message(query)
    # 모델에 대화기록을 전달하기 전에, message 타입으로 전환
    message=chat_history.message
    try:
     ai_message=model(message)
     chat_history.add_ai_message(response.content)
     print(f'챗봇: {response.content}')
    except Exception as e:
     print('챗봇: 오류 발생')
     print(f'오류: {e}')
     break

print(chat_history)