# 🧩 AI 코딩 네이밍 가이드
### Azure AI 기술 구성요소 및 아키텍처
URL: https://pro-psm-webapp-ffagfpebdpd2f9at.swedencentral-01.azurewebsites.net/
---

## 🔹 개요  
이 애플리케이션은 **RAG (Retrieval-Augmented Generation)** 패턴을 사용하여 개발 과정에서 발생하는 명명 규칙, 용어 정의, 일반적인 Q&A 요청에 대해 전문가 수준의 답변을 제공합니다.  

---

## ⚙️ 사용 Azure 기술 구성요소

### 🖥️ 1. **Azure Web App**
- **프론트엔드 및 백엔드 RAG 로직의 통합 주체.**
- Streamlit을 통한 사용자 인터페이스 제공 및 요청 처리.
- **키워드 추출, 임베딩 생성, Azure AI Search 검색, 최종 LLM 답변 생성** 등 모든 핵심 RAG 파이프라인 로직을 이 애플리케이션 내에서 직접 실행합니다.

### 🧠 2. **Azure OpenAI**
- **LLM (Large Language Model) 모델 (`GPT-4.1-mini`)**
    - 사용자 요청에서 핵심 키워드를 추출.
    - 검색된 Context를 기반으로 최종 전문가 답변을 생성.
- **Embedding 모델 (`text-embedding-3-small`)**
    - 사용자 요청을 벡터로 변환하여 벡터 검색에 사용.

### 🔍 3. **Azure AI Search**
- **RAG Context 저장 및 검색 엔진.**
- **하이브리드 검색 (키워드 + 벡터)**을 통해 관련성이 높은 Context를 검색합니다.
- **활용 인덱스:**
    - 명명 규칙 Index (Rules)
    - 용어사전 Index (Dictionary)
    - Q&A Index

### 💾 4. **Azure Storage Account**
- **RAG 지식 기반 (Knowledge Base) 원본 데이터 저장소.**
- 명명 규칙, 용어사전, Q&A 등을 포함하는 **학습 데이터 (Source Data)** 파일들을 안전하게 저장 및 관리하는 데 사용.

---

## 🔁 RAG 아키텍처 흐름 요약

1️⃣ **[입력]** 사용자가 **Streamlit UI**에 질문 입력.

2️⃣ **[분석]** **OpenAI LLM** 및 **Embedding 모델**을 통해 키워드와 벡터를 추출.

3️⃣ **[검색]** **Azure AI Search**에서 추출된 키워드와 벡터를 사용하여 3가지 인덱스를 **하이브리드 검색**하여 Context를 확보.

4️⃣ **[생성]** 확보된 Context와 사용자 요청을 **OpenAI LLM**에 전달하여 최종 전문가 답변을 생성.

5️⃣ **[출력]** 최종 답변을 **Streamlit UI**에 표시.