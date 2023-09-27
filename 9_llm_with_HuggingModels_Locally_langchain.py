from langchain.llms import HuggingFacePipeline
from langchain import LLMChain, PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM


# Encoder-Decoder
def load_text2text_model():
    model_id = 'google/flan-t5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=128
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


# Decoder Only
def load_text_generation_model():
    model_id = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


local_llm = load_text2text_model()  # text2text-generation
# local_llm = load_text2text_model()  # text-generation

# print(local_llm('What is the capital of England? '))  # or ---->

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt,
                     llm=local_llm)

question = "What is the capital of England?"

print(llm_chain.run(question))
