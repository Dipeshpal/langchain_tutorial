from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI()
chat_model = ChatOpenAI()


def llm_openai_model():
    c = 0
    while c != 4:
        inp = input("Say something: ")
        ans = llm.predict(inp)
        print(ans)
        c += 1


def llm_openai_chat_model():
    c = 0
    while c != 4:
        inp = input("Say something: ")
        ans = chat_model.predict(inp)
        print(ans)
        c += 1


if __name__ == "__main__":
    print("Using OpenAI Model...")
    llm_openai_model()
    print("Using OpenAI Chat Model...")
    llm_openai_chat_model()
