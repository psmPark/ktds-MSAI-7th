import streamlit as st
import os
import logging
from dotenv import load_dotenv

# Azure 서비스 관련 라이브러리
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# ======================================================================
# 환경 변수 설정 및 클라이언트 초기화
# ======================================================================
load_dotenv()
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 환경 변수 로드
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENAI_DEPLOYMENT_MODEL = os.getenv("OPENAI_DEPLOYMENT_MODEL")
OPENAI_DEPLOYMENT_EMBEDDING = os.getenv("OPENAI_DEPLOYMENT_EMBEDDING")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME_RULES = os.getenv("AZURE_SEARCH_INDEX_NAME_RULES")
AZURE_SEARCH_INDEX_NAME_QA = os.getenv("AZURE_SEARCH_INDEX_NAME_QA")
AZURE_SEARCH_INDEX_NAME_DICT = os.getenv("AZURE_SEARCH_INDEX_NAME_DICT")
VECTOR_PROFILE_NAME = "qa-vector-profile"

# 필수 환경 변수 검사
if not all(
    [
        OPENAI_ENDPOINT,
        OPENAI_KEY,
        AZURE_SEARCH_ENDPOINT,
        AZURE_SEARCH_API_KEY,
        OPENAI_DEPLOYMENT_EMBEDDING,
    ]
):
    st.error(
        "필수 환경 변수 (OpenAI/Search)가 설정되지 않았습니다. .env 파일을 확인하십시오."
    )
    st.stop()

# 클라이언트 초기화
try:
    openai_client = AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version="2024-12-01-preview",
    )
    search_credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)

    search_client_rules = SearchClient(
        AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME_RULES, search_credential
    )
    search_client_qa = SearchClient(
        AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME_QA, search_credential
    )
    search_client_dict = SearchClient(
        AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME_DICT, search_credential
    )

except Exception as e:
    st.error(f"클라이언트 초기화 오류: {e}")
    st.stop()


# ======================================================================
# RAG 파이프라인 함수
# ======================================================================


def generate_embedding(text: str) -> list:
    """텍스트를 Azure OpenAI 임베딩 모델로 변환합니다."""
    try:
        response = openai_client.embeddings.create(
            input=text, model=OPENAI_DEPLOYMENT_EMBEDDING
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"임베딩 API 호출 오류: {e}")
        return []


def extract_keywords_with_llm(user_request: str) -> tuple:
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


def search_dictionary_for_terms(user_request: str, search_query: str) -> list:
    """하이브리드 검색을 사용하여 용어사전 인덱스에서 정의된 용어를 검색합니다."""
    try:
        query_vector = generate_embedding(user_request)

        vector_queries = []
        if query_vector:
            vector_queries = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=5,
                    fields="vector_embedding",
                    exhaustive=False,
                )
            ]

        results = search_client_dict.search(
            search_text=search_query,
            vector_queries=vector_queries,
            select=["korean", "english", "abbreviation", "description"],
            top=5,
            query_type=QueryType.FULL,
        )

        dictionary_context = []
        for result in results:
            score = result.get("@search.score") or 0.0
            context = f"[Context: 용어사전(Score:{score:.2f})] **한국어**: {result.get('korean', 'N/A')} **영문**: {result.get('english', 'N/A')} **약어**: {result.get('abbreviation', 'N/A')} **설명**: {result.get('description', 'N/A')}"
            dictionary_context.append(context)
        return dictionary_context
    except Exception as e:
        logging.error(f"Azure AI Search (Dictionary Hybrid) 오류: {e}")
        return []


def search_rules_for_context(user_request: str, search_query: str) -> list:
    """하이브리드 검색을 사용하여 명명 규칙 인덱스에서 Context를 검색합니다."""
    try:
        query_vector = generate_embedding(user_request)

        vector_queries = []
        if query_vector:
            vector_queries = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=5,
                    fields="vector_embedding",
                    exhaustive=False,
                )
            ]

        results = search_client_rules.search(
            search_text=search_query,
            vector_queries=vector_queries,
            select=["category", "type", "rule_en", "rule_kr", "example"],
            top=5,
            query_type=QueryType.FULL,
        )

        context_list = []
        for result in results:
            score = result.get("@search.score") or 0.0
            examples = result.get("example", [])
            example_str = ", ".join(examples) if examples else "예시 없음"
            context = f"[Context: {result.get('category', 'N/A')} {result.get('type', 'N/A')} Rule (Score:{score:.2f})] **규칙**: {result.get('rule_kr', 'N/A')} **예시**: {example_str}"
            context_list.append(context)
        return context_list
    except Exception as e:
        logging.error(f"Azure AI Search (Rules Hybrid) 오류: {e}")
        return []


def search_qa_for_context(user_request: str, search_query: str) -> list:
    """하이브리드 검색을 사용하여 Q&A 인덱스에서 Context를 검색합니다."""
    try:
        query_vector = generate_embedding(user_request)

        vector_queries = []
        if query_vector:
            vector_queries = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=3,
                    fields="vector_embedding",
                    exhaustive=False,
                )
            ]

        results = search_client_qa.search(
            search_text=search_query,
            vector_queries=vector_queries,
            select=["category", "question", "answer"],
            top=3,
            query_type=QueryType.FULL,
        )

        context_list = []
        for result in results:
            score = result.get("@search.score") or 0.0
            context = f"[Context: QA-{result.get('category', 'N/A')} (Score:{score:.2f})] **질문**: {result.get('question', 'N/A')} **답변**: {result.get('answer', 'N/A')}"
            context_list.append(context)
        return context_list
    except Exception as e:
        logging.error(f"Azure AI Search (QA Hybrid) 오류: {e}")
        return []


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


# ======================================================================
# Streamlit UI 구성
# ======================================================================

# Session State 초기화
if "run_rag" not in st.session_state:
    st.session_state.run_rag = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "show_warning" not in st.session_state:
    st.session_state.show_warning = False
if "history" not in st.session_state:
    st.session_state.history = []
if "show_warning_empty" not in st.session_state:
    st.session_state.show_warning_empty = False
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "show_result" not in st.session_state:
    st.session_state.show_result = False


def set_example_query(query):
    """예시 질문을 설정하는 콜백 함수"""
    st.session_state.user_input = query
    st.session_state.run_rag = True
    st.session_state.current_result = None


def start_rag_process():
    """RAG 파이프라인을 실행하는 콜백 함수"""
    if st.session_state.is_processing:
        st.session_state.show_warning = True
        return

    if st.session_state.user_input:
        st.session_state.run_rag = True
        st.session_state.is_processing = True
        st.session_state.current_result = None
    else:
        st.session_state.show_warning_empty = True


def load_history_result(history_index):
    """기록된 결과를 로드하는 콜백 함수"""
    st.session_state.current_result = st.session_state.history[history_index]
    st.session_state.user_input = st.session_state.history[history_index]["question"]
    st.session_state.show_result = True


# 페이지 설정
st.set_page_config(page_title="MVP RAG 기반 명명/용어 가이드", layout="wide")

st.title("RAG 기반 명명/용어 생성 전문가")
st.markdown("---")

# 사이드바 구성
with st.sidebar:
    st.header("개요")
    st.info(
        "RAG 기반 명명/용어 생성 전문가는 Azure AI Search와 Azure OpenAI를 활용한 지능형 코딩 명명 규칙 도우미 애플리케이션입니다. "
        "개발자들이 일관된 코딩 표준을 준수하면서 변수명, 함수명, 데이터베이스 객체명 등을 생성할 수 있도록 지원합니다."
    )
    st.markdown("---")

    # 예시 질문 버튼
    st.header("예시 질문")
    st.button(
        "Java 변수명 규칙 질문(1)",
        on_click=set_example_query,
        args=["Java에서 '재고'를 나타내는 변수명을 규칙에 맞게 만들어줘."],
        key="example1",
    )
    st.button(
        "Java 변수명 규칙 질문(2)",
        on_click=set_example_query,
        args=["Java에서 배열 변수명으로 user_list는 규칙에 맞나요?"],
        key="example2",
    )
    st.button(
        "DB 인덱스 규칙 질문",
        on_click=set_example_query,
        args=["두 개의 컬럼에 걸친 복합 인덱스를 명명하는 규칙을 알려줘."],
        key="example3",
    )
    st.button(
        "용어 정의 요청",
        on_click=set_example_query,
        args=[
            "WebUI에서 '배송 준비중' 상태를 표시하는 라벨의 접두어와 해당 용어의 약어를 알려줘."
        ],
        key="example4",
    )

    st.markdown("---")

    # 검색 기록 표시
    st.header("검색 기록")
    if st.session_state.history:
        for i, item in enumerate(st.session_state.history[::-1]):
            actual_index = len(st.session_state.history) - 1 - i
            st.button(
                f"질문: {item['question'][:30]}...",
                key=f"hist_{actual_index}",
                on_click=load_history_result,
                args=[actual_index],
            )
    else:
        st.info("아직 검색 기록이 없습니다.")

# 메인 영역 구성
user_input_area = st.text_area(
    "명명 규칙, 용어 정의 또는 새로운 용어 생성을 요청하세요:",
    key="user_input",
    placeholder="예: '재고'를 나타내는 변수명을 Java 규칙에 맞게 만들어주고, '재고'의 약어는 무엇인지 알려줘.",
    height=150,
)

run_button = st.button(
    "전문가 답변 생성",
    type="primary",
    disabled=st.session_state.is_processing,
    on_click=start_rag_process,
)

# 경고 메시지 표시
if st.session_state.get("show_warning"):
    st.warning("이미 답변을 생성 중입니다. 잠시 기다려 주십시오.")
    st.session_state.show_warning = False

if st.session_state.get("show_warning_empty"):
    st.warning("요청 내용을 입력해 주세요.")
    st.session_state.show_warning_empty = False

# RAG 실행 로직
if st.session_state.run_rag and st.session_state.user_input:
    with st.spinner("전문가 답변과 Context를 검색하고 있습니다..."):
        try:
            final_user_input = st.session_state.user_input

            # 키워드 추출
            keywords_list, search_query = extract_keywords_with_llm(final_user_input)

            # Context 검색
            rules_context = search_rules_for_context(final_user_input, search_query)
            dictionary_context = search_dictionary_for_terms(
                final_user_input, search_query
            )
            qa_context = search_qa_for_context(final_user_input, search_query)

            # 모든 Context 통합
            all_context = rules_context + dictionary_context + qa_context

            if not all_context:
                final_answer = (
                    "죄송합니다. 관련된 명명 규칙이나 용어를 찾을 수 없습니다."
                )
            else:
                # 최종 응답 생성
                final_answer = generate_response_with_llm(final_user_input, all_context)

            # 결과를 세션 상태에 저장
            result_data = {
                "question": final_user_input,
                "answer": final_answer,
                "metadata": {
                    "검색_쿼리": search_query,
                    "키워드": keywords_list,
                    "총_검색된_Context_수": len(all_context),
                },
                "rules_context": "\n".join(rules_context) if rules_context else "",
                "dictionary_context": (
                    "\n".join(dictionary_context) if dictionary_context else ""
                ),
                "qa_context": "\n".join(qa_context) if qa_context else "",
            }

            # 기록에 추가 및 현재 결과 설정
            st.session_state.history.append(result_data)
            st.session_state.current_result = result_data
            st.session_state.show_result = True

        except Exception as e:
            st.error(f"파이프라인 실행 중 오류가 발생했습니다. 상세 오류: {e}")
        finally:
            # 처리 완료 후 플래그 초기화
            st.session_state.is_processing = False
            st.session_state.run_rag = False
            # 페이지 재실행으로 사이드바 업데이트 및 결과 표시
            st.rerun()

# 결과 표시
if st.session_state.show_result and st.session_state.current_result:
    result = st.session_state.current_result

    st.success("답변 생성 완료")
    st.markdown("### 최종 답변")
    st.info(result["answer"])

    # 모든 Context 정보를 기본적으로 접힌 Expander 내부에 배치
    with st.expander("상세 검색 Context 및 메타데이터", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(
            ["요약 정보", "1. 명명 규칙", "2. 용어사전", "3. Q&A"]
        )
        with tab1:
            st.markdown("##### 검색 메타데이터")
            st.json(result["metadata"])
        with tab2:
            st.markdown("##### Rules Index (Hybrid Search 결과)")
            st.code(
                result["rules_context"] if result["rules_context"] else "Context 없음",
                language="markdown",
            )
        with tab3:
            st.markdown("##### Dictionary Index (Hybrid Search 결과)")
            st.code(
                (
                    result["dictionary_context"]
                    if result["dictionary_context"]
                    else "Context 없음"
                ),
                language="markdown",
            )
        with tab4:
            st.markdown("##### Q&A Index (Hybrid Search 결과)")
            st.code(
                result["qa_context"] if result["qa_context"] else "Context 없음",
                language="markdown",
            )
