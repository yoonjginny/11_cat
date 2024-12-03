# FastAPI 서버 코드
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rag  # rag.py를 불러옵니다

app = FastAPI()

# 요청 모델 정의 (쿼리를 받기 위한 모델)
class QueryRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []  # 대화 내역 필드 추가

@app.post("/query")
async def query(request: QueryRequest):
    # 요청에서 쿼리 텍스트와 대화 내역을 가져옴
    query_text = request.query
    history = request.history
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query is required")

    # rag.py에서 정의된 generate_response 함수 호출
    try:
        result = rag.generate_response(query_text, history)
        print("Generated response:", result) #**
    except Exception as e:
        print("Error in FastAPI:", str(e)) #**
        #if 'items' in str(e): 
        #    raise HTTPException(status_code=404, detail="Items not found in response")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    # 결과를 반환
    return {"result": result}

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# api 터미널 입력 코드
# uvicorn main:app --reload