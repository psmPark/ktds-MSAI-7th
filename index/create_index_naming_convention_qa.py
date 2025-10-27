import os
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from azure.search.documents.indexes.models import (
    SearchableField,
    SimpleField,
    SearchFieldDataType,
    SearchIndex,
)
from azure.core.credentials import AzureKeyCredential

# 1. 환경 변수 로드
load_dotenv()


INDEX_NAME = "qna-convention-index"
search_endpoint = os.getenv("SEARCH_ENDPOINT")
search_api_key = os.getenv("SEARCH_API_KEY")

credential = AzureKeyCredential(search_api_key)

index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

# Q&A 데이터를 위한 인덱스 스키마 정의
qna_convention_fields = [
    # 1. ID 필드 (Key) - Edm.String 타입, Key=True는 필수
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        sortable=True,
        filterable=True,  # 숫자 ID를 문자열로 변환하여 저장할 경우, 필터링 가능하도록 설정
    ),
    # 2. Category (주제 분류) - WebUI, Java, Database 필터링/패싯 가능
    SearchableField(
        name="category",
        type=SearchFieldDataType.String,
        facetable=True,
        filterable=True,
        sortable=True,
        analyzer_name="standard",  # 표준 분석기 사용
    ),
    # 3. Question (질문) - 사용자 검색 쿼리 대상
    # 한국어 분석기(ko.microsoft) 적용하여 검색 정확도 향상
    SearchableField(
        name="question",
        type=SearchFieldDataType.String,
        analyzer_name="ko.microsoft",  # 한국어 분석기 적용
    ),
    # 4. Answer (답변) - RAG Context로 제공될 내용
    # 한국어 분석기(ko.microsoft) 적용
    SearchableField(
        name="answer",
        type=SearchFieldDataType.String,
        analyzer_name="ko.microsoft",
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
    fields=qna_convention_fields,
    scoring_profiles=scoring_profiles,
    # suggesters=suggesters,
)

result = index_client.create_index(index)
