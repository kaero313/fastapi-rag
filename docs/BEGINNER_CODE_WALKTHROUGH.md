# Beginner Code Walkthrough (Line by Line)

이 문서는 **파이썬을 처음 배우는 사람**이 코드를 보면서 이해할 수 있도록,
각 파일을 **줄(라인) 단위로 설명**합니다.

> 대상: Python 서버 코드(백엔드). 프론트(HTML/CSS/JS) 설명은 필요하면 추가 가능.

---

## 1) `app/core/config.py`

1) `from pydantic_settings import BaseSettings, SettingsConfigDict`
- Pydantic 설정 클래스를 가져옵니다. 환경변수(.env)를 읽는 데 사용됩니다.

2) *(빈 줄)*
- 코드 가독성을 위한 줄바꿈입니다.

3) *(빈 줄)*
- 클래스 정의와 임포트 구분을 위한 여백입니다.

4) `class Settings(BaseSettings):`
- 환경변수를 읽는 설정 클래스를 선언합니다.

5) `gemini_api_key: str`
- 필수 설정: Gemini API Key 문자열입니다.

6) `gemini_model: str = "gemini-2.5-flash"`
- 기본 LLM 모델 이름입니다. .env에서 바꿀 수 있습니다.

7) `gemini_embedding_model: str = "text-embedding-004"`
- 임베딩(벡터) 모델 이름입니다.

8) `embed_max_retries: int = 3`
- 임베딩 호출 실패 시 재시도 횟수입니다.

9) `embed_retry_backoff: float = 1.0`
- 재시도 시 대기 시간(초)의 기본값입니다.

10) `ingest_batch_size: int = 64`
- 문서 인입 시 한 번에 처리할 문서 수입니다.

11) `ingest_chunk_size: int = 4000`
- 청킹 기준 토큰 수(근사치)입니다.

12) `ingest_chunk_overlap: int = 400`
- 청크 간 겹치는 토큰 수입니다.

13) `candidate_k_multiplier: int = 10`
- 검색 후보 개수를 `top_k * multiplier`로 늘리는 값입니다.

14) `candidate_k_min: int = 100`
- 후보 개수의 최소값입니다.

15) `chroma_persist_dir: str = "data/chroma"`
- Chroma DB가 저장되는 폴더입니다.

16) `chroma_collection: str = "rag"`
- Chroma 컬렉션 이름입니다.

17) `top_k: int = 4`
- 기본 검색 결과 개수입니다.

18) `ingest_base_dir: str = "data/ingest"`
- 디렉터리 인입 시 기준이 되는 폴더입니다.

19) *(빈 줄)*
- 아래 설정 구분용 줄바꿈입니다.

20) `model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")`
- `.env` 파일을 읽어 설정을 덮어쓰도록 합니다.

21) *(빈 줄)*
- 클래스 정의 종료 여백입니다.

22) *(빈 줄)*
- 아래 인스턴스 생성과 분리된 여백입니다.

23) `settings = Settings()`
- Settings 클래스를 실제로 생성합니다. 이제 `settings.xxx`로 설정 사용 가능합니다.

---

## 2) `app/schemas.py`

1) `from typing import Any`
- `Any`는 “어떤 타입이든 가능”이라는 의미입니다.

2) *(빈 줄)*
- 가독성을 위한 줄바꿈입니다.

3) `from pydantic import BaseModel`
- Pydantic의 기본 모델 클래스를 가져옵니다.

4) *(빈 줄)*
- 줄바꿈입니다.

5) *(빈 줄)*
- 클래스 구분용 여백입니다.

6) `class DocumentIn(BaseModel):`
- 문서 입력을 정의하는 모델입니다.

7) `id: str | None = None`
- 문서 ID(선택). 없으면 자동 생성됩니다.

8) `text: str`
- 문서 본문 텍스트입니다.

9) `metadata: dict[str, Any] | None = None`
- 메타데이터(출처, 페이지 등)를 담는 딕셔너리입니다.

10) *(빈 줄)*
- 줄바꿈입니다.

11) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

12) `class IngestRequest(BaseModel):`
- `/ingest` 요청 바디 모델입니다.

13) `documents: list[DocumentIn]`
- 여러 문서를 리스트로 받습니다.

14) *(빈 줄)*
- 줄바꿈입니다.

15) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

16) `class QueryRequest(BaseModel):`
- `/query` 요청 바디 모델입니다.

17) `query: str`
- 사용자 질문 문자열입니다.

18) `top_k: int | None = None`
- 선택적으로 검색 결과 수를 지정합니다.

19) `source: str | None = None`
- 단일 source 필터입니다.

20) `sources: list[str] | None = None`
- 복수 source 필터입니다.

21) `page_gte: int | None = None`
- 페이지 필터 최소값입니다.

22) `page_lte: int | None = None`
- 페이지 필터 최대값입니다.

23) *(빈 줄)*
- 줄바꿈입니다.

24) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

25) `class IngestDirectoryRequest(BaseModel):`
- `/ingest-dir` 요청 바디 모델입니다.

26) `directory: str`
- 인입할 디렉터리 경로(기본 폴더 기준)입니다.

27) `recursive: bool = True`
- 하위 폴더까지 탐색할지 여부입니다.

28) `extensions: list[str] | None = None`
- 특정 확장자만 인입하고 싶을 때 사용합니다.

29) *(빈 줄)*
- 줄바꿈입니다.

30) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

31) `class CountTokensRequest(BaseModel):`
- `/count-tokens` 요청 모델입니다.

32) `text: str`
- 토큰 수를 알고 싶은 텍스트입니다.

33) `model: str | None = None`
- 토큰 카운트에 사용할 모델 이름입니다.

34) *(빈 줄)*
- 줄바꿈입니다.

35) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

36) `class Source(BaseModel):`
- 검색 결과에 포함되는 “소스 문서” 구조입니다.

37) `id: str`
- 문서(청크) ID입니다.

38) `score: float`
- 유사도 점수입니다.

39) `text: str`
- 문서 내용 일부입니다.

40) `metadata: dict[str, Any] | None = None`
- 메타데이터입니다.

41) *(빈 줄)*
- 줄바꿈입니다.

42) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

43) `class QueryResponse(BaseModel):`
- `/query` 응답 모델입니다.

44) `answer: str`
- LLM이 생성한 답변입니다.

45) `sources: list[Source]`
- 근거로 사용된 소스 목록입니다.

46) *(빈 줄)*
- 줄바꿈입니다.

47) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

48) `class CountTokensResponse(BaseModel):`
- `/count-tokens` 응답 모델입니다.

49) `model: str`
- 토큰 카운트를 계산한 모델 이름입니다.

50) `tokens: int`
- 계산된 토큰 수입니다.

51) *(빈 줄)*
- 줄바꿈입니다.

52) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

53) `class SourcesResponse(BaseModel):`
- `/sources` 응답 모델입니다.

54) `sources: list[str]`
- 저장된 source 목록입니다.

55) *(빈 줄)*
- 줄바꿈입니다.

56) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

57) `class ResetDbRequest(BaseModel):`
- `/reset-db` 요청 모델입니다.

58) `confirm: bool = False`
- 실수 방지용 확인 플래그입니다.

59) *(빈 줄)*
- 줄바꿈입니다.

60) *(빈 줄)*
- 다음 클래스 구분용 여백입니다.

61) `class ResetDbResponse(BaseModel):`
- `/reset-db` 응답 모델입니다.

62) `collection: str`
- 초기화된 컬렉션 이름입니다.

63) `before_count: int`
- 초기화 전 문서 수입니다.

64) `after_count: int`
- 초기화 후 문서 수입니다.

65) `reset: bool`
- 초기화 성공 여부입니다.

66) `errors: list[str] | None = None`
- 초기화 과정 중 오류 메시지 목록입니다.

---
