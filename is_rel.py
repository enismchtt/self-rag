from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

class IsRel(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or OllamaLLM(model="gemma3", temperature=0)

        self.prompt_template = PromptTemplate.from_template(
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "It does not need to be a stringent test. The goal is to filter out erroneous retrievals.\n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
            "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.\n\n"
            "Retrieved document:\n\n{document}\n\nUser question:\n{question}\n\nAnswer with 'yes' or 'no' only:"
        )

    def invoke(self, input: dict) -> str:
        question = input["question"]
        document = input["document"]
        prompt = self.prompt_template.format(document=document, question=question)
        response = self.llm.invoke(prompt).strip().lower()

        if "yes" in response:
            return "yes"
        elif "no" in response:
            return "no"
        else:
            return "unknown"
