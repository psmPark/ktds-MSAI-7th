import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SimpleField,
    SearchFieldDataType,
    SearchIndex,
    SearchField,  # SearchField 사용
    VectorSearch,
    HnswAlgorithmConfiguration,  # HNSW 알고리즘 설정
    VectorSearchProfile,
)
from azure.core.credentials import AzureKeyCredential

# ======================================================================
# 환경 변수 및 설정
# ======================================================================
load_dotenv()

# 인덱스 설정
INDEX_NAME = "dictionary-index"
VECTOR_DIMENSION = 1536
VECTOR_PROFILE_NAME = "dict-vector-profile"
VECTOR_ALGORITHM_NAME = "dict-hnsw-algorithm"

# Azure Search 접속 정보
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_api_key = os.getenv("AZURE_SEARCH_API_KEY")

if not all([search_endpoint, search_api_key]):
    raise ValueError("필수 환경 변수가 설정되어야 합니다.")

credential = AzureKeyCredential(search_api_key)
index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)


# ======================================================================
# A. 벡터 필드 및 Vector Search 설정 정의
# ======================================================================

# 1. 벡터 필드 정의
VECTOR_EMBEDDING_FIELD = SearchField(
    name="vector_embedding",
    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
    vector_search_dimensions=VECTOR_DIMENSION,
    vector_search_profile_name=VECTOR_PROFILE_NAME,
    searchable=True,
    retrievable=False,
)

# 2. Vector Search 설정 (알고리즘 및 프로파일)
COMMON_VECTOR_SEARCH = VectorSearch(
    profiles=[
        VectorSearchProfile(
            name=VECTOR_PROFILE_NAME, algorithm_configuration_name=VECTOR_ALGORITHM_NAME
        )
    ],
    algorithms=[
        HnswAlgorithmConfiguration(
            name=VECTOR_ALGORITHM_NAME,
            kind="hnsw",
            parameters={
                "m": 4,
                "ef_construction": 400,
                "ef_search": 500,
                "metric": "cosine",
            },
        ),
    ],
)


# ======================================================================
# B. 용어사전 인덱스 필드 정의 (기존 필드 + 벡터 필드)
# ======================================================================

dictionary_fields = [
    # 1. ID 필드
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        sortable=True,
        filterable=True,
    ),
    # 2. Korean (분석기 이름을 문자열로 직접 지정)
    SearchableField(
        name="korean",
        type=SearchFieldDataType.String,
        filterable=True,
        analyzer_name="ko.microsoft",
    ),
    # 3. English (분석기 이름을 문자열로 직접 지정)
    SearchableField(
        name="english", type=SearchFieldDataType.String, analyzer_name="en.microsoft"
    ),
    # 4. Abbreviation (분석기 이름을 문자열로 직접 지정)
    SearchableField(
        name="abbreviation", type=SearchFieldDataType.String, analyzer_name="keyword"
    ),
    # 5. Description (분석기 이름을 문자열로 직접 지정)
    SearchableField(
        name="description",
        type=SearchFieldDataType.String,
        analyzer_name="ko.microsoft",
    ),
    # 6. 벡터 필드
    VECTOR_EMBEDDING_FIELD,
]


# ======================================================================
# C. 인덱스 생성 및 배포 (삭제 후 재생성)
# ======================================================================

# 인덱스 객체 생성
dictionary_index = SearchIndex(
    name=INDEX_NAME,
    fields=dictionary_fields,
    vector_search=COMMON_VECTOR_SEARCH,
)

print(f"--- Azure AI Search Index Deployment: {INDEX_NAME} ---")

# 1. 기존 인덱스 삭제
try:
    print(f"1. Deleting existing index '{INDEX_NAME}' if it exists...")
    index_client.delete_index(INDEX_NAME)
    print("   -> Existing index deleted successfully.")
except Exception:
    print("   -> Index did not exist or could not be deleted (continuing...).")

# 2. 인덱스 재생성
try:
    print(f"2. Creating new index '{INDEX_NAME}' with Vector Search configuration...")
    result = index_client.create_index(dictionary_index)
    print(f"   -> Index '{result.name}' created successfully with Vector Search.")

except Exception as e:
    print(f"   -> ERROR creating index {INDEX_NAME}: {e}")
    raise

print("--- Deployment Complete. Index is ready for vector data upload. ---")
