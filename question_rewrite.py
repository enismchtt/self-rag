from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

class QuestionRewriter(Runnable) :

    def __init__(self, llm=None):

        self.llm = llm or OllamaLLM(model="gemma3", temperature=0)


        self.prompt_template = PromptTemplate.from_template(

            "Sen, verilen bir soruyu daha iyi hale getirerek vektör veritabanı (vectorstore) sorguları için optimize eden bir soru yeniden yazma asistanısın.\n"
            "Girdi sorunun altında yatan anlamsal niyeti/sebebi anlamaya çalış ve bu bağlamda soruyu iyileştir. \n"
            "Girdi sorusu : \n\n {question} \n."
            "Sadece iyileştirilmiş haliyle yeni bir Türkçe soru üret. Soruyu üretirken girdi sorusundaki bağlamdan uzaklaşma. Soru dışında bir şey yazma.\n"
        )

        self.output_parser = StrOutputParser()

    def invoke(self, input: dict) -> str:
        question = input["question"]
        prompt = self.prompt_template.format(question=question)
        llm_response = self.llm.invoke(prompt)
        parsed_response = self.output_parser.invoke(llm_response)
        return parsed_response.strip()