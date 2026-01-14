import os
from dotenv import load_dotenv

# .env 파일의 내용을 환경변수로 로드합니다.
load_dotenv()

class Settings:
    # 프로젝트 이름
    PROJECT_NAME: str = "FastAPI Gemini RAG"
    
    # .env에서 가져온 API 키 (없으면 에러 발생 가능성 있음)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    
    # 벡터 DB가 저장될 로컬 폴더 경로
    CHROMA_DB_DIR: str = "./chroma_db"
    
    # 업로드된 파일이 임시로 저장될 폴더
    UPLOAD_DIR: str = "./uploaded_files"

# 싱글톤처럼 사용하기 위해 인스턴스 생성
settings = Settings()

# 업로드 폴더가 없으면 미리 생성 (Java의 static block이나 @PostConstruct 느낌)
if not os.path.exists(settings.UPLOAD_DIR):
    os.makedirs(settings.UPLOAD_DIR)