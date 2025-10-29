import os
import json
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import SearchField
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
INDEX_NAME = "coding-convention-index"  # 명명 규칙 인덱스 이름

# 파일 경로 (로컬에 있어야 함)
DATA_FILE_PATH = "data/naming_rules.json"  # ⭐⭐ 실제 파일 경로로 수정하세요 ⭐⭐

if not os.path.exists(DATA_FILE_PATH):
    logging.error(f"데이터 파일 경로 오류: {DATA_FILE_PATH} 파일을 찾을 수 없습니다.")
    exit()

# 클라이언트 초기화
openai_client = AzureOpenAI(
    api_key=OPENAI_KEY, azure_endpoint=OPENAI_ENDPOINT, api_version="2024-12-01-preview"
)
search_credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
search_client = SearchClient(AZURE_SEARCH_ENDPOINT, INDEX_NAME, search_credential)


# ======================================================================
# A. 임베딩 생성 함수
# ======================================================================
def generate_embedding(text: str, client: AzureOpenAI) -> list[float]:
    """텍스트를 Azure OpenAI text-embedding-3-small 모델로 임베딩합니다."""
    try:
        response = client.embeddings.create(
            input=text, model=OPENAI_DEPLOYMENT_EMBEDDING
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"임베딩 생성 실패 for text: '{text[:30]}...' 오류: {e}")
        return None


# ======================================================================
# B. 데이터 임베딩 및 인덱스 업로드 (Push)
# ======================================================================
def upload_data_with_vectors(file_path: str):
    """JSON 파일을 읽어 벡터를 생성하고 Search Index에 업로드합니다."""

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents_to_upload = []

    for doc in data:
        # 1. 임베딩할 텍스트 통합 (검색 품질을 위해 rule_kr, rule_en, example을 모두 임베딩)
        text_to_embed = f"{doc.get('rule_kr', '')} {doc.get('rule_en', '')} {' '.join(doc.get('example', []))}"

        # 2. 임베딩 벡터 생성
        vector = generate_embedding(text_to_embed, openai_client)

        if vector:
            # 3. 문서 구조 조정 및 벡터 필드 추가
            doc["id"] = str(doc["id"])  # ID는 문자열이어야 함
            doc["vector_embedding"] = vector  # 인덱스의 벡터 필드 이름과 일치

            if "id_num" in doc:
                del doc["id_num"]

            documents_to_upload.append(doc)

        else:
            logging.warning(f"문서 ID {doc.get('id')} 임베딩 건너뜀.")

    if documents_to_upload:
        print(
            f"\n--- {len(documents_to_upload)}개의 문서 임베딩 완료. Azure Search에 업로드 시작 ---"
        )
        try:
            # 4. 인덱스에 일괄 업로드 (Upload Documents)
            results = search_client.upload_documents(documents=documents_to_upload)
            logging.info(
                f"업로드 성공: {len(results)}개 문서. 첫 번째 키: {results[0].key}"
            )
        except Exception as e:
            logging.error(f"Azure Search 업로드 실패: {e}")

    else:
        print("업로드할 유효한 문서가 없습니다.")


if __name__ == "__main__":
    upload_data_with_vectors(DATA_FILE_PATH)
