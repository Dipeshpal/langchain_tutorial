from langchain.schema import BaseOutputParser


class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")


ans = CommaSeparatedListOutputParser().parse("hi, bye")
print(ans)

