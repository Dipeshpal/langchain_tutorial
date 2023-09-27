from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI(openai_api_key="...")
chat_model = ChatOpenAI()


def llm_openai_model():
    c = 0
    while c != 2:
        inp = input("Say something: ")
        ans = llm.predict(inp)
        print(ans)
        c += 1


def llm_openai_chat_model():
    c = 0
    while c != 2:
        inp = input("Say something: ")
        ans = chat_model.predict(inp)
        print(ans)
        c += 1


if __name__ == "__main__":
    llm_openai_model()
    llm_openai_chat_model()
