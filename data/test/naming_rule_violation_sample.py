from openai import AzureOpenAI
from dotenv import load_dotenv
import os

py_version = "3.10"
llm_model = "dev-gpt-4.1-mini"

load_dotenv()

API_KEY_STR = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT_STR = os.getenv("AZURE_ENDPOINT")
API_VERSION_STR = os.getenv("OPENAI_API_VERSION")

pyClient = AzureOpenAI(
    api_key=API_KEY_STR,
    azure_endpoint=AZURE_ENDPOINT_STR,
    api_version=API_VERSION_STR,
)


def run_poem_generator():
    """시 생성기를 실행하는 메인 함수입니다."""
    
    i = 0
    while True:
        i += 1
        
        sub = input(f"[{i}차 시도] 시의 주제를 입력하세요: ")

        if sub.lower() == "exit":
            break

        cont = input("시의 내용을 입력하세요: ")
        
        msg = [
            {"role": "system", "content": "You are a AI poem."},
            {
                "role": "user",
                "content": f"주제: {sub}, 내용: {cont}로 시를 작성해줘.",
            },
        ]

        try:
            resp = pyClient.chat.completions.create(
                model=llm_model, messages=msg, temperature=0.8
            )

            print("-" * 100)
            print(resp.choices[0].message.content)
            
        except Exception as err_msg: 
            print("오류 발생: ", err_msg)


if __name__ == "__main__":
    run_poem_generator()