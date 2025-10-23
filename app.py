# chatbot_app.py
import streamlit as st
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

# Azure AI Search 설정
search_endpoint = "https://<YOUR_SEARCH_SERVICE>.search.windows.net"
search_index = "<YOUR_INDEX_NAME>"
search_key = "<YOUR_SEARCH_KEY>"
search_client = SearchClient(endpoint=search_endpoint,
                             index_name=search_index,
                             credential=AzureKeyCredential(search_key))

# Azure OpenAI 설정
openai_client = OpenAI(api_key="<YOUR_OPENAI_KEY>")

# --- 헬퍼 함수 ---
def search_internal_docs(query, top_k=3):
    results = search_client.search(query, top=top_k, query_type=QueryType.SIMPLE)
    snippets = []
    for r in results:
        snippets.append(r['content'])
    return snippets

def generate_gpt_answer(query, context_snippets):
    context_text = "\n".join(context_snippets)
    prompt = f"Context를 바탕으로 질문에 답변해 주세요.\nContext:\n{context_text}\nQuestion: {query}\nAnswer:"
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_abbreviation(full_name):
    prompt = f"'{full_name}'에 대한 짧고 표준화된 약어를 생성하고 설명도 포함해 주세요."
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content.strip()

def classify_intent(user_query):
    prompt = f"""
사용자 질문을 다음 의도로 분류하세요:
1. 내부 용어/약어 검색
2. 신규 약어 생성
번호만 답변해주세요.
사용자 질문: "{user_query}"
"""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}]
    )
    intent = response.choices[0].message.content.strip()
    if intent == "1":
        return "search"
    elif intent == "2":
        return "abbreviation"
    else:
        return "unknown"

# --- Streamlit UI ---
st.title("내부 지식 챗봇")

user_input = st.text_input("질문을 입력하거나 약어 생성을 요청하세요:")
if st.button("제출") and user_input:
    intent = classify_intent(user_input)
    
    if intent == "search":
        snippets = search_internal_docs(user_input)
        if snippets:
            answer = generate_gpt_answer(user_input, snippets)
            st.markdown(f"**답변:** {answer}")
            st.markdown("**출처 문서 스니펫:**")
            for s in snippets:
                st.write(f"- {s}")
        else:
            st.write("관련 문서를 찾을 수 없습니다.")
    
    elif intent == "abbreviation":
        abbrev = generate_abbreviation(user_input)
        st.markdown(f"**추천 약어:** {abbrev}")
    
    else:
        st.write("죄송합니다. 요청을 이해하지 못했습니다.")
