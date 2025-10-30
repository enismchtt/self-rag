from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

class IsSup(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or OllamaLLM(model="gemma3", temperature=0)

        self.prompt_template = PromptTemplate.from_template(
            "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n"
            "Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts.\n\n"
            "Set of facts:\n\n{documents}\n\n"
            "LLM generation:\n\n{generation}\n\n"
            "Answer with 'yes' or 'no' only:"
        )

    def invoke(self, input: dict) -> str:
        documents = input["documents"]
        generation = input["generation"]
        prompt = self.prompt_template.format(documents=documents, generation=generation)
        response = self.llm.invoke(prompt).strip().lower()

        if "yes" in response:
            return "yes"
        elif "no" in response:
            return "no"
        else:
            return "unknown"
