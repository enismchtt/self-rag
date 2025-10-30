from typing import List
from typing_extensions import TypedDict

#  typed dict to check langgraph state structure goes as planned . Done for ide suggestions

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retry_count: int  # Counts number of retrieval can't exceed 2 for each graph run