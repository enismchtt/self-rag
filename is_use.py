# IsUse token generation :  Does the generation answers the question ? 

from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

class IsUse(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or OllamaLLM(model="gemma3", temperature=0)

        self.prompt_template = PromptTemplate.from_template(
            "You are a grader assessing whether an answer addresses / resolves a question \n"
            "User question: \n\n {question} \n\n LLM generation: {generation}\n\n"
            "Answer with 'yes' or 'no' only:"
        )

    def invoke(self, input: dict) -> str:
        question = input["question"]
        generation = input["generation"]
        prompt = self.prompt_template.format(question=question, generation=generation)
        response = self.llm.invoke(prompt).strip().lower()

        if "yes" in response:
            return "yes"
        elif "no" in response:
            return "no"
        else:
            return "unknown"
