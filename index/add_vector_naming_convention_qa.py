import os
import json
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# ======================================================================
# 환경 변수 및 설정
# ======================================================================
load_dotenv()
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 환경 변수 재사용
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENAI_DEPLOYMENT_EMBEDDING = os.getenv("OPENAI_DEPLOYMENT_EMBEDDING")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
INDEX_NAME = "qna-convention-index"  # ⭐⭐ QA 인덱스 이름으로 변경 ⭐⭐

# 파일 경로
DATA_FILE_PATH = (
    "data/naming_convention_qa.json"  # ⭐⭐ QA 데이터 파일 경로로 변경 ⭐⭐
)

if not os.path.exists(DATA_FILE_PATH):
    logging.error(f"데이터 파일 경로 오류: {DATA_FILE_PATH} 파일을 찾을 수 없습니다.")
    exit()

# 클라이언트 초기화
try:
    openai_client = AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version="2024-12-01-preview",
    )
    search_credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    search_client = SearchClient(AZURE_SEARCH_ENDPOINT, INDEX_NAME, search_credential)
except Exception as e:
    logging.error(f"클라이언트 초기화 오류: {e}")
    exit()


# ======================================================================
# A. 임베딩 생성 함수
# ======================================================================
def generate_embedding(text: str, client: AzureOpenAI) -> list[float]:
    """텍스트를 Azure OpenAI text-embedding-3-small 모델로 임베딩합니다."""
    if not text or len(text.strip()) < 5:
        return None

    try:
        response = client.embeddings.create(
            input=text, model=OPENAI_DEPLOYMENT_EMBEDDING
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"임베딩 생성 실패 for text: '{text[:50]}...' 오류: {e}")
        return None


# ======================================================================
# B. 데이터 임베딩 및 인덱스 업로드 (Push)
# ======================================================================
def upload_data_with_vectors(file_path: str):
    """JSON 파일을 읽어 벡터를 생성하고 Search Index에 업로드합니다."""

    print(f"데이터 파일 로드 시작: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents_to_upload = []

    for idx, doc in enumerate(data):

        if not isinstance(doc, dict):
            logging.warning(
                f"문서 #{idx+1}의 타입이 'dict'가 아닙니다 (타입: {type(doc).__name__}). 건너뜁니다."
            )
            continue

        # 1. 임베딩할 텍스트 통합 (QA Index의 키에 맞게 수정)
        # 키: 'question', 'answer', 'category' 사용
        text_to_embed = (
            f"질문: {doc.get('question', '')}. "
            f"답변: {doc.get('answer', '')}. "
            f"카테고리: {doc.get('category', '')}"
        ).strip()

        # 2. 임베딩 벡터 생성
        vector = generate_embedding(text_to_embed, openai_client)

        if vector:
            # 3. 문서 구조 조정 및 벡터 필드 추가
            doc["id"] = str(doc.get("id", idx + 1))
            doc["vector_embedding"] = vector  # 인덱스의 벡터 필드 이름과 일치

            documents_to_upload.append(doc)

        else:
            logging.warning(
                f"문서 ID {doc.get('id', idx + 1)} 임베딩 건너뜀 (텍스트 부족 또는 API 오류)."
            )

    if documents_to_upload:
        print(
            f"\n--- {len(documents_to_upload)}개의 문서 임베딩 완료. Azure Search에 업로드 시작 ({INDEX_NAME}) ---"
        )
        try:
            results = search_client.upload_documents(documents=documents_to_upload)
            success_count = sum(1 for res in results if res.succeeded)
            logging.info(
                f"업로드 완료: {success_count}개 성공, {len(results) - success_count}개 실패."
            )
        except Exception as e:
            logging.error(f"Azure Search 업로드 실패: {e}")

    else:
        print("업로드할 유효한 문서가 없습니다.")


if __name__ == "__main__":
    upload_data_with_vectors(DATA_FILE_PATH)
