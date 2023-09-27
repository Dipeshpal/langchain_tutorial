# pip -q install langchain huggingface_hub transformers sentence_transformers accelerate bitsandbytes
import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HUGGINGFACEHUB_API_TOKEN'

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# you can use  Encoder-Decoder Model ("text-generation") or  Encoder-Decoder Model ("text2text-generation")
llm_chain = LLMChain(prompt=prompt,
                     llm=HuggingFaceHub(repo_id="google/flan-t5-xl",
                                        model_kwargs={"temperature":0,
                                                      "max_length":64}))

question = "What is the capital of England?"

print(llm_chain.run(question))
