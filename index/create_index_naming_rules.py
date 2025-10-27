import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchableField,
    SimpleField,
    SearchFieldDataType,
)
from azure.core.credentials import AzureKeyCredential

# 1. 환경 변수 로드
load_dotenv()

INDEX_NAME = "coding-convention-index"
search_endpoint = os.getenv("SEARCH_ENDPOINT")
search_api_key = os.getenv("SEARCH_API_KEY")

credential = AzureKeyCredential(search_api_key)

index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

# RAG 파이프라인에서 문서 검색을 위한 인덱스 스키마 정의
# 인덱스명: 'coding-convention-index-v2' (예시)

coding_convention_fields = [
    # 1. ID 필드 (Key) - Int를 String으로 변환하여 사용
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        sortable=True,  # ID로 정렬 가능하도록 설정
    ),
    # 2. Category (대분류)
    SearchableField(
        name="category",
        type=SearchFieldDataType.String,
        facetable=True,
        filterable=True,
        sortable=True,
        analyzer_name="standard",  # 대소문자 구분 없는 일반 검색에 적합
    ),
    # 3. Type (유형)
    SearchableField(
        name="type",
        type=SearchFieldDataType.String,
        facetable=True,
        filterable=True,
        sortable=True,
        analyzer_name="standard",
    ),
    # 4. rule_en (영문 규칙 원문) - 영어 분석기 사용
    SearchableField(
        name="rule_en",
        type=SearchFieldDataType.String,
        analyzer_name="en.microsoft",  # 또는 "en.lucene"
    ),
    # 5. rule_kr (한국어 규칙 설명) - 한국어 분석기 사용
    SearchableField(
        name="rule_kr",
        type=SearchFieldDataType.String,
        analyzer_name="ko.microsoft",  # 또는 "ko.lucene"
    ),
    # 6. example (예시 코드 목록) - Collection(String) 타입
    SearchableField(
        name="example",
        collection=True,
        type=SearchFieldDataType.String,
        analyzer_name="standard",  # 예시 코드의 일반적인 검색
    ),
    # 7. id_num (선택 사항: 원본 정수 ID) - 정수형으로 필터링/정렬이 필요할 경우
    SimpleField(
        name="id_num",
        type=SearchFieldDataType.Int32,  # 원본 데이터 타입 사용
        filterable=True,
        sortable=True,
    ),
]

# 이제 이 'coding_convention_fields' 리스트를 사용하여 Index를 생성할 수 있습니다.

scoring_profiles = []
suggesters = [
    {
        "name": "sg",
        "sourceFields": ["Address/City", "Address/Country", "Tags"],
        "searchMode": "analyzingInfixMatching",
    }
]

index = SearchIndex(
    name=INDEX_NAME,
    fields=coding_convention_fields,
    scoring_profiles=scoring_profiles,
    # suggesters=suggesters,
)

result = index_client.create_index(index)
