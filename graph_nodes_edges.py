
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph, START

from retriever import Retriever
from is_rel import IsRel
from is_sup import IsSup
from is_use import IsUse
from generator import Generator
from question_rewrite import QuestionRewriter



class Graph: 

    def __init__(self , llm , docs ):


        self.retriever = Retriever(doc_list=docs)
        self.is_rel = IsRel(llm=llm)
        self.is_sup = IsSup(llm=llm)
        self.is_use = IsUse(llm=llm)

        self.generator = Generator(llm=llm)
        self.q_rewriter = QuestionRewriter(llm=llm)





    ### Nodes


    def retrieve(self,state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.retrieve_docs(question)  # at max 5 
        return {"documents": documents, "question": question}


    def generate(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.generator.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION (IsRel)---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            isrel = self.is_rel.invoke(
                {"question": question, "document": d.page_content}
            )
           
            if isrel == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


    def transform_query(self,state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        retry_count = state.get("retry_count", 0) + 1

        # Re-write question
        better_question = self.q_rewriter.invoke({"question": question})  # Do not uses documents while generating new question
        return {"documents": documents, "question": better_question, "retry_count": retry_count}


    ### Edges


    def decide_to_generate(self,state):
        """
        Determines whether to generate an answer, or re-generate a question. 

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        #If relevant documents founded -> Generate   else regenerate question

        if not filtered_documents: # there exist some docs that relevant
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"


    def stop_if_limit_reached(self,state):  # if number of retrivals above 2  finish 
        return "exceeded" if state.get("retry_count", 0) >= 2 else "retrieve"



    def grade_generation_v_documents_and_question(self,state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS (IsSup -> IsUse)---")  # IsSup  and IsUse calls 
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        is_sup = self.is_sup.invoke(
            {"documents": documents, "generation": generation}
        )
     

        # Check hallucination
        if is_sup == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            is_use = self.is_use.invoke({"question": question, "generation": generation})
          
            if is_use == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-GENERATING---")  # pprint option avail.
            return "not supported"