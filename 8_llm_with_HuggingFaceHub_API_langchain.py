# pip -q install langchain huggingface_hub transformers sentence_transformers accelerate bitsandbytes
import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv

load_dotenv()
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HUGGINGFACEHUB_API_TOKEN'

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = HuggingFaceHub(repo_id="databricks/dolly-v2-3b",
                     model_kwargs={"temperature": 0.5, "max_length": 64})

# you can use  Encoder-Decoder Model ("text-generation") or  Encoder-Decoder Model ("text2text-generation")
llm_chain = LLMChain(prompt=prompt,
                     llm=llm)

question = "What is the capital of England?"

print(llm_chain.run(question))
