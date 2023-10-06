from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.llms import OpenAI
from langchain.tools import BaseTool
from dotenv import load_dotenv
import datetime


load_dotenv()


class CustomTool(BaseTool):
    name = "Datetime"
    description = "useful when you need to answer current datetime"

    def _run(self, query: str) -> str:
        dt = datetime.datetime.now()
        return dt.strftime("%Y-%m-%d %H:%M")

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("not supported")


llm = OpenAI(model_name='text-davinci-003', temperature=0)

tools = [CustomTool()]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

ans = agent.run('What is the current time?')

print(ans)
