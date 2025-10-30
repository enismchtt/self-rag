from retriever import Retriever

# Example usage:
if __name__ == "__main__":
    # Initialize retriever
    retriever = Retriever(doc_list=["docs/7.5.41250.pdf",
        "docs/7.5.42333.pdf",
        "docs/7.5.42375.pdf"])
    
    # Query for similar documents
    results = retriever.retrieve_docs("Maarif Arşivi ve Müzeler Daire Başkanlığında Daire Başkanlığının görevleri nelerdir ? ", k=5)
    
    # Print results
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content}")
        print(f"Relevance: {doc.metadata.get('relevance', 'N/A')}")
        print("------------------------------------------------------------------")
        print("\n\n")

    
