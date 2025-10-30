from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

class Generator(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or OllamaLLM(model="gemma3", temperature=0)

        self.prompt_template = PromptTemplate.from_template(
            "Sen bir soru-cevap asistanısın. "
            "Aşağıdaki bağlama göre soruyu cevapla. "
            "Cevabını sadece Türkçe olarak ver. Eğer bağlamda bilgi yoksa 'Bilmiyorum' de. "
            "En fazla üç cümle kullan ve cevabı öz tut.\n\n"
            "Bağlam:\n{context}\n\n"
            "Soru:\n{question}\n\n"
            "Cevap:"
        )

        self.parser = StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.parser

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, input: dict) -> str:
        # Expecting input = {"question": str, "docs": list of documents}
        question = input["question"]
        context = self.format_docs(input["context"])
        
        result = self.chain.invoke({"context": context, "question": question})
        return result.strip()
