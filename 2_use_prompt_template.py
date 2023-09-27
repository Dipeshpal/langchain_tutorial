# Example of PromptTemplate
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
print(prompt.format(product="colorful socks"))

# Example of ChatPromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

print(chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming."))
