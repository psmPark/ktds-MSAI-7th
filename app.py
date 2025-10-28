import streamlit as st
import os
import json
import logging
import re
from dotenv import load_dotenv

# OpenAI 라이브러리 (pip install openai)
from openai import AzureOpenAI

# Azure Search 라이브러리 (pip install azure-search-documents)
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# 로깅 설정
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 1. 환경 변수 로드
load_dotenv()

# ======================================================================
# 환경 변수 설정 (로컬 .env 파일에서 로드)
# ======================================================================
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENAI_DEPLOYMENT_MODEL = os.getenv("OPENAI_DEPLOYMENT_MODEL")
OPENAI_DEPLOYMENT_EMBEDDING = os.getenv("OPENAI_DEPLOYMENT_EMBEDDING")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME_RULES = os.getenv("AZURE_SEARCH_INDEX_NAME_RULES")
AZURE_SEARCH_INDEX_NAME_QA = os.getenv("AZURE_SEARCH_INDEX_NAME_QA")
AZURE_SEARCH_INDEX_NAME_DICT = os.getenv("AZURE_SEARCH_INDEX_NAME_DICT")

# logging.info(f"OPENAI_ENDPOINT: {OPENAI_ENDPOINT}")
# logging.info(f"OPENAI_KEY: {OPENAI_KEY}")
# logging.info(f"OPENAI_DEPLOYMENT_MODEL: {OPENAI_DEPLOYMENT_MODEL}")
# logging.info(f"OPENAI_DEPLOYMENT_EMBEDDING: {OPENAI_DEPLOYMENT_EMBEDDING}")
# logging.info(f"AZURE_SEARCH_ENDPOINT: {AZURE_SEARCH_ENDPOINT}")
# logging.info(f"AZURE_SEARCH_API_KEY: {AZURE_SEARCH_API_KEY}")
# logging.info(f"AZURE_SEARCH_INDEX_NAME_RULES: {AZURE_SEARCH_INDEX_NAME_RULES}")
# logging.info(f"AZURE_SEARCH_INDEX_NAME_QA: {AZURE_SEARCH_INDEX_NAME_QA}")
# logging.info(f"AZURE_SEARCH_INDEX_NAME_DICT: {AZURE_SEARCH_INDEX_NAME_DICT}")

# 1. 필수 환경 변수 검사
if not all(
    [
        OPENAI_ENDPOINT,
        OPENAI_KEY,
        AZURE_SEARCH_ENDPOINT,
        AZURE_SEARCH_API_KEY,
        OPENAI_DEPLOYMENT_EMBEDDING,
    ]
):
    logging.error(
        "필수 환경 변수 (OpenAI 또는 Search)가 설정되지 않았습니다. .env 파일을 확인하십시오."
    )
    exit()

# 2. OpenAI 클라이언트 초기화
openai_client = AzureOpenAI(
    api_key=OPENAI_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version="2024-12-01-preview",
)

# 3. Azure Search 클라이언트 초기화
search_credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
search_client_rules = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME_RULES,
    credential=search_credential,
)
search_client_qa = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME_QA,
    credential=search_credential,
)
search_client_dict = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME_DICT,
    credential=search_credential,
)


# ======================================================================
# RAG 파이프라인 함수 (이전 Function App 로직과 동일)
# ======================================================================


# 💡 임베딩 생성 함수 추가
def generate_embedding(text: str) -> list[float]:
    """텍스트를 Azure OpenAI text-embedding-3-small 모델로 임베딩합니다."""
    try:
        response = openai_client.embeddings.create(
            input=text, model=OPENAI_DEPLOYMENT_EMBEDDING
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"임베딩 API 호출 오류: {e}")
        return []


# 1. 키워드 추출 함수
def extract_keywords_with_llm(user_request: str) -> tuple[list, str]:
    """GPT 모델을 사용하여 사용자 요청에서 핵심 키워드 리스트와 검색 쿼리 문자열을 반환합니다."""
    try:
        prompt = f"""
        다음 사용자 요청에서 명명 규칙 및 용어 검색에 필요한 핵심 키워드(Key Term)와 카테고리(Java, Database, WebUI)를
        최대 5개까지 쉼표로 구분하여 추출하세요. 요청: {user_request} ->
        """
        response = openai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for keyword extraction.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        keywords_str = response.choices[0].message.content.strip()
        keywords_list = [k.strip() for k in keywords_str.split(",") if k.strip()]
        search_query = " OR ".join(keywords_list)
        return keywords_list, search_query
    except Exception as e:
        logging.error(f"OpenAI 키워드 추출 오류: {e}")
        return [user_request], user_request


# 2. 용어사전 검색 함수 (Dictionary Index) - 💡 하이브리드 검색으로 수정
def search_dictionary_for_terms(user_request: str, search_query: str) -> list:
    """하이브리드 검색(키워드 + 벡터)을 사용하여 용어사전 인덱스에서 정의된 용어를 검색합니다."""
    try:
        # 1. 임베딩 벡터 생성
        query_vector = generate_embedding(user_request)

        if not query_vector:
            logging.warning("임베딩 벡터 생성 실패. 키워드 검색만 수행합니다.")
            vector_queries = []
        else:
            # 2. VectorizedQuery 객체 생성
            vector_queries = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=5,  # 검색할 K개의 이웃 수
                    fields="vector_embedding",  # 인덱스 내 벡터 필드 이름
                    exhaustive=False,  # HNSW 알고리즘 사용
                )
            ]

        # 3. 하이브리드 검색 실행 (search_text와 vector_queries 동시 사용)
        results = search_client_dict.search(
            search_text=search_query,  # ⬅️ 키워드 검색 (FULL Lucene)
            vector_queries=vector_queries,  # ⬅️ 벡터 검색
            select=["korean", "english", "abbreviation", "description"],
            top=5,  # 상위 5개 결과 반환
            query_type=QueryType.FULL,  # 키워드 검색 타입
        )

        dictionary_context = []
        for result in results:
            # Score를 포함하여 검색 품질 확인 가능
            context = f"[Context: 용어사전(Score:{result['@search.score']:.2f})] **한국어**: {result['korean']} **영문**: {result['english']} **약어**: {result['abbreviation']} **설명**: {result['description']}"
            dictionary_context.append(context)
        return dictionary_context

    except Exception as e:
        logging.error(f"Azure AI Search (Dictionary Hybrid) 오류: {e}")
        return []


# 3. 명명 규칙 Context 검색 함수 (Rules Index) - 💡 하이브리드 검색으로 수정
def search_rules_for_context(user_request: str, search_query: str) -> list:
    """하이브리드 검색(키워드 + 벡터)을 사용하여 명명 규칙 인덱스에서 Context를 검색합니다."""
    try:
        # 1. 임베딩 벡터 생성
        query_vector = generate_embedding(user_request)

        if not query_vector:
            logging.warning("Rules: 임베딩 벡터 생성 실패. 키워드 검색만 수행합니다.")
            vector_queries = []
        else:
            # 2. VectorizedQuery 객체 생성
            vector_queries = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=5,  # 검색할 K개의 이웃 수 (최대 5개)
                    fields="vector_embedding",  # 인덱스 내 벡터 필드 이름
                    exhaustive=False,
                )
            ]

        # 3. 하이브리드 검색 실행 (search_text와 vector_queries 동시 사용)
        results = search_client_rules.search(
            search_text=search_query,  # ⬅️ 키워드 검색
            vector_queries=vector_queries,  # ⬅️ 벡터 검색
            select=["category", "type", "rule_en", "rule_kr", "example"],
            top=5,  # 검색 결과를 조금 더 늘려 Context 확보 (기존 3개에서 5개로 변경)
            query_type=QueryType.FULL,
        )

        context_list = []
        for result in results:
            context = f"[Context: {result['category']} {result['type']} Rule (Score:{result['@search.score']:.2f})] **규칙**: {result['rule_kr']} **예시**: {', '.join(result['example'])}"
            context_list.append(context)
        return context_list

    except Exception as e:
        logging.error(f"Azure AI Search (Rules Hybrid) 오류: {e}")
        return []


# 4. Q&A Context 검색 함수 (QA Index) - 💡 하이브리드 검색으로 수정
def search_qa_for_context(user_request: str, search_query: str) -> list:
    """하이브리드 검색(키워드 + 벡터)을 사용하여 Q&A 인덱스에서 Context를 검색합니다."""
    try:
        # 1. 임베딩 벡터 생성
        query_vector = generate_embedding(user_request)

        if not query_vector:
            logging.warning("QA: 임베딩 벡터 생성 실패. 키워드 검색만 수행합니다.")
            vector_queries = []
        else:
            # 2. VectorizedQuery 객체 생성
            vector_queries = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=3,  # 검색할 K개의 이웃 수 (최대 3개)
                    fields="vector_embedding",
                    exhaustive=False,
                )
            ]

        # 3. 하이브리드 검색 실행
        results = search_client_qa.search(
            search_text=search_query,  # ⬅️ 키워드 검색
            vector_queries=vector_queries,  # ⬅️ 벡터 검색
            select=["category", "question", "answer"],
            top=3,  # 검색 결과를 조금 더 늘려 Context 확보 (기존 2개에서 3개로 변경)
            query_type=QueryType.FULL,
        )

        context_list = []
        for result in results:
            context = f"[Context: QA-{result['category']} (Score:{result['@search.score']:.2f})] **질문**: {result['question']} **답변**: {result['answer']}"
            context_list.append(context)
        return context_list

    except Exception as e:
        logging.error(f"Azure AI Search (QA Hybrid) 오류: {e}")
        return []


# 5. 최종 응답 생성 함수 (Generation)
def generate_response_with_llm(user_request: str, context_list: list) -> str:
    """검색된 Context를 활용하여 사용자 요청에 대한 최종 답변을 생성합니다."""
    context_str = "\n".join(context_list)

    system_prompt = f"""
    당신은 코딩 명명 규칙을 준수하는 전문가입니다. 사용자의 요청에 따라 새로운 변수명이나 함수명을 생성하고,
    명명 규칙 또는 용어에 대한 질문에 답변해야 합니다.
    
    **반드시 지켜야 할 사항:**
    1. 아래 '검색된 규칙', '용어사전', 'Q&A'를 참고하여 답변을 생성합니다.
    2. 새로운 명명을 생성할 경우, 해당되는 규칙(예: camelCase, PascalCase, 동사 시작 등)을 간결하게 설명합니다.
    3. 용어사전에서 찾은 영문 대안이나 약어를 활용하여 생성된 용어의 명확성을 높입니다.
    4. Q&A 컨텍스트에서 직접적인 답변을 찾았다면, 해당 내용을 중심으로 답변의 정확도를 높입니다.

    **검색된 규칙 및 용어 (Context):**
    ---
    {context_str}
    ---
    """

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_request},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI 최종 응답 생성 오류: {e}")
        return f"요청 처리 중 오류가 발생했습니다. (오류: {e})"


# ----------------------------------------------------------------------
# 🌐 메인 실행 블록 (테스트 실행)
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # --- 테스트 케이스 정의 ---
    test_requests = [
        # "Web에서 드롭다운 규칙은 뭐야?",
        "Web에서 '출발지'을 표시하는 라벨은 뭐라고 해야해?",
        # "Web에서 '계약 조건에 동의' 체크박스를 명명하는 예시는?",
    ]

    print("========================================================")
    print("      ✅ extract_keywords_with_llm 함수 테스트 시작      ")
    print("========================================================")

    for i, req in enumerate(test_requests):
        print(f"\n--- TEST CASE {i+1} ---")
        print(f"INPUT: {req}")

        # 1. 키워드 추출
        keywords_list, search_query = extract_keywords_with_llm(req)

        # 2. Context 검색 (3가지 소스)
        rules_context = search_rules_for_context(req, search_query)
        dictionary_context = search_dictionary_for_terms(req, search_query)
        qa_context = search_qa_for_context(req, search_query)

        # 3. 모든 Context 통합
        all_context = rules_context + dictionary_context + qa_context
        if not all_context:
            final_answer = "죄송합니다. 관련된 명명 규칙이나 용어를 찾을 수 없습니다. 인덱스 구성을 확인하거나 다른 질문을 시도해 보세요."
        else:
            # 4. 최종 응답 생성 (Generation)
            final_answer = generate_response_with_llm(req, all_context)
        print(f"FINAL ANSWER:\n{final_answer}")
        print("\n========================================================")
        print("                테스트 완료                           ")
        print("========================================================")
    print("\n✅ 모든 테스트 케이스가 완료되었습니다.")
