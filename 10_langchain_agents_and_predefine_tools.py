from langchain.agents import initialize_agent, AgentType
from langchain.agents import load_tools
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()


llm = OpenAI(model_name='text-davinci-003', temperature=0)

tool_name = ['llm-math']

tools = load_tools(tool_name, llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

ans = agent.run('What is 2*5')

print(ans)
