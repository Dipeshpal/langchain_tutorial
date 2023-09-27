from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain

# Model: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
model_name_or_path = "meta-llama/Llama-2-7b-hf"  # or local path: './model_file/'
model_file = 'llama-2-7b-chat.ggmlv3.q2_K.bin'
local_files_only = False
llm = CTransformers(model=model_name_or_path, model_file=model_file, local_files_only=local_files_only)

template = """
        [INST] 
        <<SYS>>
        Answer the following questions as best you can-
        <</SYS>>
        User Prompt: {user_input_query}
        [/INST]
        """

prompt = PromptTemplate(template=template, input_variables=["user_input_query"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
response = llm_chain.run("How are you?")
print(response)
