# main.py
from fastapi import FastAPI
from dotenv import load_dotenv
import os

# 환경변수 로드 (.env 파일 읽기)
load_dotenv()

app = FastAPI(title="Gemini RAG Server")

@app.get("/")
def read_root():
    return {"message": "Hello, RAG World!", "framework": "FastAPI"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# 실행을 위한 코드 (터미널에서 uvicorn 명령어로 실행해도 되지만, 디버깅을 위해 추가)
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0"은 외부 접속 허용, reload=True는 코드 수정 시 자동 재시작
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)