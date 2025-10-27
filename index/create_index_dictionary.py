import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SimpleField,
    SearchFieldDataType,
    SearchIndex,
    LexicalAnalyzerName,
)
from azure.core.credentials import AzureKeyCredential


# 1. 환경 변수 로드
load_dotenv()

INDEX_NAME = "dictionary-index"
search_endpoint = os.getenv("SEARCH_ENDPOINT")
search_api_key = os.getenv("SEARCH_API_KEY")

credential = AzureKeyCredential(search_api_key)

index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)


# 용어사전 데이터를 위한 인덱스 스키마 정의
dictionary_fields = [
    # 1. ID 필드 (Key)
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        sortable=True,
        filterable=True,
    ),
    # 2. Korean (한국어 용어)
    SearchableField(
        name="korean",
        type=SearchFieldDataType.String,
        filterable=True,  # 한국어 용어 기준으로 필터링/정렬이 필요할 수 있음
        analyzer_name="ko.microsoft",
    ),
    # 3. English (영문 용어)
    SearchableField(
        name="english",
        type=SearchFieldDataType.String,
        analyzer_name="en.microsoft",  # 영어 분석기 적용
    ),
    # 4. Abbreviation (약어) - 대소문자 그대로 검색하기 위해 keyword 분석기 사용
    SearchableField(
        name="abbreviation",
        type=SearchFieldDataType.String,
        # 약어는 완전 일치 검색이 중요하므로 Standard 대신 Keyword 분석기를 사용해 토큰화 방지
        analyzer_name="keyword",
    ),
    # 5. Description (설명)
    SearchableField(
        name="description",
        type=SearchFieldDataType.String,
        analyzer_name="ko.microsoft",
    ),
]


# 이제 이 'dictionary_fields' 리스트를 사용하여 Index를 생성할 수 있습니다.

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
    fields=dictionary_fields,
    scoring_profiles=scoring_profiles,
    # suggesters=suggesters,
)

result = index_client.create_index(index)
