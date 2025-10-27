# 🧩 개발용 약어 생성기 (Developer Abbreviation Generator) MVP  
### Azure AI 기술 구성요소 및 아키텍처 (수정본)

---

## 🔹 개요  
이 앱은 변수명, 함수명, 버튼명, 테이블명, 컬럼명 등 **개발 과정에서 필요한 약어를 자동으로 생성하고 관리**하는 시스템입니다.  
Azure OpenAI와 Azure AI Search를 결합하여, 사용자의 자연어 입력을 기반으로 일관된 약어를 추천하고 의미 기반 검색을 수행합니다.

---

## ⚙️ 사용 Azure 기술 구성요소

### 🧠 1. **Azure OpenAI Service**
- GPT-4 / GPT-4o 모델을 사용하여 약어 생성 및 명명 규칙 추천 수행  
- 자연어 이해 및 약어 생성 로직의 핵심 역할 담당  

---

### ☁️ 2. **Azure Function App**
- 백엔드 API 역할 수행  
- Streamlit 프론트엔드에서 전달된 요청을 검증하고 키워드를 추출하여 Azure AI Search 및 OpenAI API 호출을 처리  

---


### 📄 3. **Azure Storage (Blob 또는 Table Storage)**
- 생성된 약어, 사용자 요청 내역, 환경 설정 등의 데이터를 저장  
- 향후 약어 추천 시 과거 데이터 참조 가능  

---

### 🔍 4. **Azure AI Search (인덱스 튜닝 포함)**
- 약어 검색 및 유사 용어 탐색의 핵심 엔진  
- **Semantic Search**, **Vector Search**, **Scoring Profile**, **Field Weight 조정** 등을 활용하여 검색 정확도 향상  
- **Relevance 튜닝(점수 조정 및 Boost)** 을 통해 결과 품질을 지속적으로 최적화  

---

### 🖥️ 5. **Azure Web App / Container App**
- Streamlit 기반 UI를 Azure에 배포하여 웹 인터페이스 제공  
- 개발자가 약어를 입력하고 검색 결과를 즉시 확인 가능  

---

## 🔁 아키텍처 흐름 요약

1️⃣ 사용자가 **Streamlit UI**에서 약어 요청 입력  
2️⃣ **Function App**이 요청을 수신 및 검증  
3️⃣ **Azure AI Search**에서 인덱스 튜닝된 점수 기반으로 유사 용어 검색  
4️⃣ **Azure OpenAI**가 최종 약어 생성 및 보정 수행  
5️⃣ 결과를 **Streamlit UI**에 표시
