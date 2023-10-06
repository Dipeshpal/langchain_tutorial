from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from load_llm_models import *
from llm_tools import *

# load any model
# llm = load_llama_model(model_name_or_path='./model_file/', model_file='llama-2-7b-chat.ggmlv3.q6_K.bin',
#                        local_files_only=True)
llm = load_openai_llm_model()

tools = [Time(), Date(), Calculator(), CircumferenceTool(), Joke(), Camera(), AiAnswer()]

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools. Note use your brain only if tools are not relevant to questions:"""
suffix = """Begin!"
{chat_history}
Question: {input}
{agent_scratchpad}
"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

while True:
    inp = input("Enter: ")
    if inp == "q":
        break
    ans = agent_chain.run(input=inp)
    print("AI Answer: ", ans)
