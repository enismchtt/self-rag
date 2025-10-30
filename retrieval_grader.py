from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from vespa.application import Vespa
from langchain_huggingface import HuggingFaceEmbeddings

# Direct query approach with fixes
def query_vespa_directly(query_text, top_k=5):
    vespa_app = Vespa(url="http://localhost", port=8080)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    
    query_embedding = embedding_function.embed_query(query_text)

    response = vespa_app.query(
        yql=f"select * from sources * where userQuery() limit {top_k}",  
        query=query_text,  
        ranking="hybrid", 
        timeout=3000,
        ranking_features={
            "query(query_embedding)": query_embedding  
        }
    )
    
    return response.hits

# Grading prompt template
system = """You are a grader assessing relevance of a retrieved document to a user question.
Seek for matching phrases, keywords or semantic matching. The goal is to filter out erroneous retrievals.
If the document contains phrases, keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

human_template = "Retrieved document:\n\n{document}\n\nUser question:\n{question}\n\nAnswer with 'yes' or 'no' only:"

# Initialize LLM (Ollama)
llm = OllamaLLM(model="gemma3", temperature=0) # you select the model you like 

# Grading function using LLM
def grade_document(question: str, document: str) -> str:
    prompt = f"{system}\n\n{human_template.format(document=document, question=question)}"
    response = llm.invoke(prompt)
    answer = response.strip().lower()
    if "yes" in answer:
        return "yes"
    elif "no" in answer:
        return "no"
    else:
        return "unknown"

# Query and grade
"""question = "Enter your query here "
results = query_vespa_directly(question, top_k=2)

print(f"\n🔍 Grading top {len(results)} Vespa results for: \"{question}\"")

for i, hit in enumerate(results, 1):
    doc_txt = hit["fields"]["text"]
    relevance_score = hit.get("relevance", "N/A")
    grade = grade_document(question, doc_txt)
    
    print(f"\nResult {i} (Vespa score: {relevance_score}):\n✅ Relevant? {grade}\n📄 Snippet: {doc_txt[:300]}...\n" + "-"*80)
"""