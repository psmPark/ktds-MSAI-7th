import azure.functions as func
import logging
import json
import os
import re

# OpenAI 라이브러리 (pip install openai)
from openai import AzureOpenAI

# Azure Search 라이브러리 (pip install azure-search-documents)
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType

# ======================================================================
# 환경 변수 설정 (Function App 설정에 반드시 저장해야 합니다.)
# ======================================================================
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
OPENAI_DEPLOYMENT = os.environ.get("OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")

SEARCH_ENDPOINT = os.environ.get("AZURE_AI_SEARCH_ENDPOINT")
SEARCH_KEY = os.environ.get("AZURE_AI_SEARCH_API_KEY")
SEARCH_INDEX_NAME_RULES = os.environ.get(
    "AZURE_AI_SEARCH_INDEX_NAME_RULES", "coding-convention-index"
)  # 명명 규칙 인덱스
SEARCH_INDEX_NAME_QA = os.environ.get(
    "AZURE_AI_SEARCH_INDEX_NAME_QA", "qna-convention-index"
)  # Q&A 인덱스


# Azure OpenAI 클라이언트 초기화
# NOTE: Azure Function App 환경 변수에 OPENAI_ENDPOINT, OPENAI_KEY 등을 설정해야 합니다.
openai_client = AzureOpenAI(
    api_key=OPENAI_KEY, azure_endpoint=OPENAI_ENDPOINT, api_version="2024-02-15-preview"
)


# ----------------------------------------------------------------------
# 📚 용어사전 데이터 로드 (Source 2)
# ----------------------------------------------------------------------
DICTIONARY_FILE_PATH = os.path.join(os.path.dirname(__file__), "dictionary.json")

# 용어사전 JSON 배열 구조 예시 (실제 파일이 없는 경우 사용)
SAMPLE_DICTIONARY_ARRAY = [
    {
        "korean": "상품",
        "english": "Product",
        "abbreviation": "PROD",
        "description": "소비자에게 판매하는 모든 종류의 물품.",
    },
    {
        "korean": "재고",
        "english": "Stock/Inventory",
        "abbreviation": "INV",
        "description": "현재 판매 또는 배송 가능한 상태로 창고에 보관 중인 상품 수량.",
    },
    {
        "korean": "품절",
        "english": "Sold Out",
        "abbreviation": "OOS",
        "description": "재고가 모두 소진되어 일시적 또는 영구적으로 구매할 수 없는 상태. (Out Of Stock)",
    },
    {
        "korean": "할인",
        "english": "Discount",
        "abbreviation": "DC",
        "description": "정가에서 일정 비율이나 금액을 빼주는 판매 방식.",
    },
    {
        "korean": "정가",
        "english": "Regular Price",
        "abbreviation": "RP",
        "description": "할인이나 프로모션이 적용되지 않은 본래의 판매 가격.",
    },
]


def load_dictionary():
    """용어사전 파일을 로드하여 한국어 용어를 키로 하는 딕셔너리로 변환합니다."""
    try:
        if os.path.exists(DICTIONARY_FILE_PATH):
            with open(DICTIONARY_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            logging.warning(
                f"용어사전 파일 {DICTIONARY_FILE_PATH}을 찾을 수 없습니다. 샘플 데이터를 사용합니다."
            )
            data = SAMPLE_DICTIONARY_ARRAY

        # [ { "korean": "상품", ... } ] -> { "상품": { "english": "Product", ... } } 형태로 변환
        converted_dict = {item["korean"]: item for item in data}
        return converted_dict

    except Exception as e:
        logging.error(f"용어사전 로드 또는 변환 오류: {e}")
        return {}


DICTIONARY_DATA = load_dictionary()


# ----------------------------------------------------------------------
# 1. 키워드 추출 함수 (LLM 활용 - Retrieval Query 생성)
# ----------------------------------------------------------------------
def extract_keywords_with_llm(user_request: str) -> tuple[list, str]:
    """GPT 모델을 사용하여 사용자 요청에서 핵심 키워드 리스트와 검색 쿼리 문자열을 반환합니다."""
    try:
        prompt = f"""
        다음 사용자 요청에서 명명 규칙 및 용어 검색에 필요한 핵심 키워드(Key Term)와 카테고리(Java, Database, WebUI)를
        최대 5개까지 쉼표로 구분하여 추출하세요.

        요청: {user_request} ->
        """

        response = openai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
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


# ----------------------------------------------------------------------
# 2. 용어사전 검색 함수 (Retrieval Source 2)
# ----------------------------------------------------------------------
def search_dictionary_for_terms(keywords: list) -> list:
    """추출된 키워드를 기반으로 용어사전 파일에서 정의된 용어를 검색합니다."""
    dictionary_context = []

    for term_kr, term_data in DICTIONARY_DATA.items():
        is_match = any(
            keyword.lower() in term_kr.lower() or term_kr.lower() in keyword.lower()
            for keyword in keywords
        )

        if is_match:
            context = f"[Context: 용어사전] **한국어**: {term_kr} **영문**: {term_data.get('english', 'N/A')} **약어**: {term_data.get('abbreviation', 'N/A')} **설명**: {term_data.get('description', 'N/A')}"
            if context not in dictionary_context:
                dictionary_context.append(context)

    return dictionary_context


# ----------------------------------------------------------------------
# 3. Azure AI Search 호출 및 Context 검색 함수 (Retrieval Source 1: 명명 규칙)
# ----------------------------------------------------------------------
def search_rules_for_context(search_query: str) -> list:
    """명명 규칙 인덱스에서 Context를 검색합니다."""
    try:
        search_client = SearchClient(
            SEARCH_ENDPOINT, SEARCH_INDEX_NAME_RULES, SEARCH_KEY
        )

        results = search_client.search(
            search_query,
            select=["category", "type", "rule_en", "rule_kr", "example"],
            top=3,
            query_type=QueryType.FULL,
        )

        context_list = []
        for result in results:
            context = f"[Context: {result['category']} {result['type']} Rule] **규칙**: {result['rule_kr']} **예시**: {', '.join(result['example'])}"
            context_list.append(context)

        return context_list

    except Exception as e:
        logging.error(f"Azure AI Search (Rules) 오류: {e}")
        return []


# ----------------------------------------------------------------------
# 4. Azure AI Search 호출 및 Context 검색 함수 (Retrieval Source 3: Q&A)
# ----------------------------------------------------------------------
def search_qa_for_context(search_query: str) -> list:
    """Q&A 인덱스에서 Context를 검색합니다."""
    try:
        search_client = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX_NAME_QA, SEARCH_KEY)

        results = search_client.search(
            search_query,
            select=["category", "question", "answer"],
            top=2,
            query_type=QueryType.FULL,
        )

        context_list = []
        for result in results:
            context = f"[Context: QA-{result['category']}] **질문**: {result['question']} **답변**: {result['answer']}"
            context_list.append(context)

        return context_list

    except Exception as e:
        logging.error(f"Azure AI Search (QA) 오류: {e}")
        return []


# ----------------------------------------------------------------------
# 5. 최종 응답 생성 함수 (Generation)
# ----------------------------------------------------------------------
def generate_response_with_llm(user_request: str, context_list: list) -> str:
    """검색된 Context를 활용하여 사용자 요청에 대한 최종 답변을 생성합니다."""
    context_str = "\n".join(context_list)

    # System Prompt는 LLM에게 명확한 역할과 Context 활용 지침을 부여합니다.
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
            model=OPENAI_DEPLOYMENT,
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
# 🌐 메인 Function App 엔트리 포인트 (통합된 RAG 로직)
# ----------------------------------------------------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    # 1. 요청 수신 및 검증
    try:
        req_body = req.get_json()
        user_request = req_body.get("request_text")
        if not user_request:
            raise ValueError("요청 본문에 'request_text' 필드가 필요합니다.")
    except Exception as e:
        return func.HttpResponse(str(e), status_code=400)

    # 2. 핵심 키워드 추출
    keywords_list, search_query = extract_keywords_with_llm(user_request)

    # 3. 명명 규칙 Context 검색 (Source 1)
    rules_context = search_rules_for_context(search_query)

    # 4. 용어사전에서 관련 용어 Context 검색 (Source 2)
    dictionary_context = search_dictionary_for_terms(keywords_list)

    # 5. Q&A Context 검색 (Source 3)
    qa_context = search_qa_for_context(search_query)

    # 6. 모든 Context를 통합
    all_context = rules_context + dictionary_context + qa_context
    logging.info(f"통합된 Context 개수: {len(all_context)}")

    # 7. Azure OpenAI가 최종 생성 및 보정 수행 (Generation)
    final_answer = generate_response_with_llm(user_request, all_context)

    # 8. 결과 반환
    response_data = {
        "final_answer": final_answer,
        "search_query_used": search_query,
        "retrieved_context_count": len(all_context),
    }

    return func.HttpResponse(
        json.dumps(response_data, ensure_ascii=False),
        mimetype="application/json",
        status_code=200,
    )
