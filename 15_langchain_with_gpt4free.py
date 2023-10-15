from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
# import gpt4free
# from gpt4free import Provider
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import g4f


class EducationalLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        out = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[{"role": "user", "content": prompt}],
        )  #
        if stop:
            stop_indexes = (out.find(s) for s in stop if s in out)
            min_stop = min(stop_indexes, default=-1)
            if min_stop > -1:
                out = out[:min_stop]
        return out


llm = EducationalLLM()

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}? Just tell one and only the name",
)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("colorful socks"))
