
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from vespa.application import Vespa
from vespa.package import ApplicationPackage, Document as VespaDocument, Field, RankProfile, Schema, QueryProfile, QueryProfileType, QueryTypeField, HNSW
from vespa.deployment import VespaDocker
from typing import List

# Retrieval Based on query using Vespa
class Retriever:
    def __init__(self, doc_list: List[str], app_name="selfrag"):
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
        self.doc_list = doc_list
        self.app_name = app_name
  
        # Get embedding dimension
        sample_embedding = self.embedding_function.embed_query("sample text")
        self.embedding_dim = len(sample_embedding)
        
        # Connect to deployed app
        self.vespa_app = Vespa(url="http://localhost", port=8080)
        
        # Index documents
        self._index_documents()

    
    def _load_documents(self):
        docs = []
        for doc_path in self.doc_list:
            try:
                docs.extend(PyPDFLoader(doc_path).load())
                print(f" Loaded {doc_path}")
            except Exception as e:
                print(f"Error loading {doc_path}: {e}")
        return docs

    def _split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=25,
            separators=[r"\nMADDE \d+", r"\nBÖLÜM", "\n\n", "\n", " "]
        )
        return text_splitter.split_documents(docs)

    def _index_documents(self):
        docs = self._load_documents()
        doc_splits = self._split_documents(docs)
        print(f" Created {len(doc_splits)} text chunks")

        for idx, doc in enumerate(doc_splits):
            embedding = self.embedding_function.embed_query(doc.page_content)
            
            # Convert embedding to proper tensor format for Vespa
            tensor_embedding = {str(i): val for i, val in enumerate(embedding)}
            
            try:
                result = self.vespa_app.feed_data_point(
                    schema=self.app_name,
                    data_id=str(idx),
                    fields={
                        "text": doc.page_content,
                        "embedding": tensor_embedding,
                    },
                )
                if idx % 50 == 0:
                    print(f" Indexed {idx + 1} documents")
            except Exception as e:
                print(f" Error indexing document {idx}: {e}")

    def retrieve_docs(self, query: str, k=5):
        """Retrieve documents using semantic similarity"""
        query_embedding = self.embedding_function.embed_query(query)
        
        # Convert to tensor format (list of floats)
        tensor_embedding = [float(x) for x in query_embedding]
        
        try:
            # Use nearestNeighbor search operator for semantic similarity
            response = self.vespa_app.query(body={
                'yql': f'select * from sources {self.app_name} where {{targetNumHits:{k}}}nearestNeighbor(embedding,query_embedding)',
                'hits': k,
                'ranking.features.query(query_embedding)': tensor_embedding,
                'ranking.profile': 'semantic-similarity'
            })
            
            docs = []
            for hit in response.hits:
                fields = hit.get("fields", {})
                if "text" in fields:
                    docs.append(Document(
                        page_content=fields["text"],
                        metadata={
                            "relevance": hit.get("relevance", 0),
                            "document_id": hit.get("id", "")
                        }
                    ))
            
            return docs
            
        except Exception as e:
            print(f" Error during retrieval: {e}")
            return []

