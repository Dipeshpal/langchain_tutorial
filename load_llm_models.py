from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFacePipeline, CTransformers, OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate, LLMChain, HuggingFaceHub

from dotenv import load_dotenv

load_dotenv()


def load_openai_llm_model():
    llm = OpenAI()
    return llm


def load_openai_llm_chat_model():
    chat_model = ChatOpenAI()
    return chat_model


def load_hugging_face_llm_model(repo_id="google/flan-t5-xl", model_kwargs=None):
    """
    you can use  Encoder-Decoder Model ("text-generation") or  Encoder-Decoder Model ("text2text-generation")
    """
    template = """Question: {question}
    
    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    if model_kwargs is None:
        model_kwargs = {"temperature": 0, "max_length": 64}
    llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs))
    return llm_chain


# Encoder-Decoder
def load_text2text_model(template: str = None, input_variables: list = None,
                         model_id='google/flan-t5-small', load_in_8bit=True,
                         device_map='auto', max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=load_in_8bit, device_map=device_map)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm
    # if template is None:
    #     template = """Question: {question}
    #        Answer: Let's think step by step."""
    #
    # if input_variables is None:
    #     input_variables = ["question"]
    #
    # prompt = PromptTemplate(template=template, input_variables=input_variables)
    #
    # llm_chain = LLMChain(prompt=prompt,
    #                      llm=local_llm)
    # return llm_chain


# Decoder Only
def load_text_generation_model(template: str = None, input_variables: list = None,
                               model_id="gpt2-medium", load_in_8bit=True,
                               device_map='auto', max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=load_in_8bit, device_map=device_map)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm
    # if template is None:
    #     template = """Question: {question}
    #              Answer: Let's think step by step."""
    #
    # if input_variables is None:
    #     input_variables = ["question"]
    #
    # prompt = PromptTemplate(template=template, input_variables=input_variables)
    #
    # llm_chain = LLMChain(prompt=prompt,
    #                      llm=local_llm)
    # return llm_chain


def load_llama_model(template: str = None, input_variables: list = None,
                     model_name_or_path="meta-llama/Llama-2-7b-hf",
                     model_file='llama-2-7b-chat.ggmlv3.q2_K.bin', local_files_only=False):
    llm = CTransformers(model=model_name_or_path, model_file=model_file, local_files_only=local_files_only)
    return llm
    # if template is None:
    #     template = """
    #             [INST]
    #             <<SYS>>
    #             Answer the following questions as best you can-
    #             <</SYS>>
    #             User Prompt: {user_input_query}
    #             [/INST]
    #             """
    #
    # if input_variables is None:
    #     input_variables = ["user_input_query"]
    #
    # prompt = PromptTemplate(template=template, input_variables=input_variables)
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    # return llm_chain
