from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.llms import OpenAI
import yfinance as yf
from langchain.tools import BaseTool


class CustomTool(BaseTool):
    name = "ytfinance"
    description = "useful when you need to answer questions about the current stock price of stock ticker"

    def _run(self, query: str) -> str:
        tk = yf.Ticker(query)
        return tk.info.get('currentPrice')

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("not supported")


llm = OpenAI(model_name='text-davinci-003', temperature=0)

tools = [CustomTool()]

# tools = load_tools(tool_name, llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

ans = agent.run('What is the current price of stock ticker Nvidia?')
