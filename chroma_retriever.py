
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# Retrieval Basd on query currently works on ChromaDB
class Retriever:
    def __init__(self, doc_list, collection_name="rag-chroma"):
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
        self.doc_list = doc_list
        self.collection_name = collection_name
        self.retriever = self._setup_retriever()

    def _load_documents(self):
        docs = []
        for doc_path in self.doc_list:
            try:
                docs.extend(PyPDFLoader(doc_path).load())
                print(f"✅ Loaded {doc_path}")
            except Exception as e:
                print(f"❌ Error loading {doc_path}: {e}")
        return docs

    def _split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=25,
            separators=[r"\nMADDE \d+", r"\nBÖLÜM", "\n\n", "\n", " "]
        )
        return text_splitter.split_documents(docs)

    def _setup_retriever(self):
        docs = self._load_documents()
        doc_splits = self._split_documents(docs)
        print(f"📝 Created {len(doc_splits)} text chunks")

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=self.collection_name,
            embedding=self.embedding_function,
        )
        return vectorstore.as_retriever(search_kwargs={"k": 5}) # at most 5 retrieval 

    def retrieve_docs(self, query ):
        return self.retriever.get_relevant_documents(query)

