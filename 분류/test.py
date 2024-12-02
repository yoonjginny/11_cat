from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini")

print(model.invoke("안녕"))