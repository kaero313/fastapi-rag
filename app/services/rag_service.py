from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from app.core.config import settings # 위에서 만든 설정 import

class RAGService:
    # 생성자 (__init__): Java의 Constructor
    def __init__(self):
        # 1. 임베딩 모델 설정 (텍스트 -> 숫자 변환기)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        # 2. LLM 모델 설정 (Gemini Pro)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            temperature=0, # 0에 가까울수록 사실 기반 답변 (창의성 낮춤)
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        # 3. 벡터 DB 초기화 (기존 데이터가 있으면 로드)
        self.vector_db = Chroma(
            persist_directory=settings.CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )

    # 기능 1: PDF 파일 학습시키기 (Ingestion)
    def learn_pdf(self, file_path: str):
        # A. PDF 로드
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # B. 텍스트 쪼개기 (Chunking)
        # 문맥을 유지하며 1000자 단위로 자르고, 200자는 겹치게 함
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        
        # C. 벡터 DB에 저장 (Embed & Store)
        self.vector_db.add_documents(docs)
        
        return f"총 {len(docs)}개의 조각으로 나누어 학습을 완료했습니다."

    # 기능 2: 질문에 답변하기 (Retrieval + Generation)
    def ask_question(self, question: str):
        # DB를 검색기(Retriever) 모드로 전환
        retriever = self.vector_db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3} # 가장 유사한 문서 3개를 가져와라
        )
        
        # RAG 체인 생성 (검색 -> 프롬프트 조합 -> LLM 질의)
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # 찾은 문서를 전부 프롬프트에 'stuffing(채워넣기)' 함
            retriever=retriever,
            return_source_documents=True # 답변할 때 참고한 문서도 같이 반환
        )
        
        # 실행
        result = qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "source": [doc.page_content[:100] + "..." for doc in result["source_documents"]]
        }