from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph, START
from graph_state import GraphState
from graph_nodes_edges import Graph
from pprint import pprint

# Initialize retriever and grader
#NOTE: Here please enter paths of your local documents
docs = ["docs/7.5.41250.pdf", "docs/7.5.42333.pdf", "docs/7.5.42375.pdf"]
# Ollama LLM
llm = OllamaLLM(model="gemma3", temperature=0)  # Single llm will be used for isRel , isSup , isUse calls



workflow = StateGraph(GraphState)  # uses the defined typed dict 

graph = Graph(llm=llm , docs= docs)  # graph instances where edges and nodes defined 


# Define the nodes
workflow.add_node("retrieve", graph.retrieve)  # retrieve
workflow.add_node("grade_documents", graph.grade_documents)  # grade documents
workflow.add_node("generate", graph.generate)  # generate
workflow.add_node("transform_query", graph.transform_query)  # Re writing the question

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    graph.decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "transform_query",
    graph.stop_if_limit_reached, # If it's retrieved 2 times already stops.                  
    {
        "retrieve" : "retrieve",
        "exceeded" : END,
    },
                               
                               
)
workflow.add_conditional_edges(
    "generate",
    graph.grade_generation_v_documents_and_question,  # uses both IsUse and IsSup calls 
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)


# Compile
app = workflow.compile()



# ------------------------------------------------------------------------------------
"""
#TODO: enter your question
question = "QUESTION ACCORDING TO LOCAL FILES"

inputs = {
    "question": question,
    "retry_count": 0
}

# Streaming graph execution
for output in app.stream(inputs):
    for key, value in output.items():
        # This key is the name of the node just executed
        print(f"\n Node executed: '{key}'")
        if key == "generate" and "documents" in value:
            print("\n Retrieved Documents (used in generation):")
            for i, doc in enumerate(value["documents"]):
                try:
                    print(f"\n--- Document {i+1} ---")
                    print(doc.page_content.strip())
                except AttributeError:
                    print(f"(Document {i+1} is not a LangChain Document object)")
        

    print("\n" + "-"*60 + "\n")

# Print final answer
pprint("Final Question :")
pprint(value.get("question"))
pprint("🎯 Final Answer:")
pprint(value.get("generation", "[No Answer Generated]"))

"""





