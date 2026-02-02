import chromadb
import pandas as pd

# 절대 경로 사용 (역슬래시 \ 대신 슬래시 / 사용 추천)
db_path = "C:/project/fastapi-rag/data/chroma" 

try:
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    
    if not collections:
        print("🤔 DB 폴더는 찾았는데 컬렉션이 비어있네요.")
    else:
        for coll in collections:
            print(f"\n--- 컬렉션: {coll.name} ---")
            c = client.get_collection(coll.name)
            data = c.get(limit=3) # 딱 3개만 찍어보기
            
            df = pd.DataFrame({
                'ID': data['ids'],
                'Content': [doc[:50] + "..." for doc in data['documents']], # 긴 내용은 자름
                'Metadata': data['metadatas']
            })
            print(df)

except Exception as e:
    print(f"❌ 에러 발생: {e}")
    print("💡 팁: FastAPI 서버를 잠시 끄거나, data 폴더를 복사해서 복사본 경로로 시도해보세요!")