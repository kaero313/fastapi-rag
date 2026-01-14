from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from app.core.config import settings
from app.services.rag_service import RAGService

app = FastAPI(title="Gemini RAG Server")

# 서비스 인스턴스 생성 (Java의 @Autowired와 비슷하게 전역 변수로 사용)
rag_service = RAGService()

# DTO 정의 (Java의 Request DTO)
class QuestionRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "Server is running", "service": "Gemini RAG"}

# 1. PDF 업로드 API
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 파일 확장자 검사
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    # 파일을 서버의 지정된 폴더에 저장
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {str(e)}")
        
    # 서비스 호출: 저장된 파일을 읽어서 학습
    result_message = rag_service.learn_pdf(file_path)
    
    return {"filename": file.filename, "message": result_message}

# 2. 질문하기 API
@app.post("/ask")
def ask_gemini(request: QuestionRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="질문 내용을 입력해주세요.")
        
    response = rag_service.ask_question(request.query)
    return response

# 실행 코드 (디버깅용)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)