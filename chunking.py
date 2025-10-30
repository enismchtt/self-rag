from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter





# 4. Load and split documents
doc_list = ["docs/7.5.41250.pdf", "docs/7.5.42333.pdf", "docs/7.5.42375.pdf"]
docs = []
for doc_path in doc_list:
    try:
        docs.extend(PyPDFLoader(doc_path).load())
        print(f"✅ Loaded {doc_path}")
    except Exception as e:
        print(f"❌ Error loading {doc_path}: {e}")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=0,
    separators=[r"\nMADDE \d+", r"\nBÖLÜM", "\n\n", "\n", " "]
)

doc_splits = text_splitter.split_documents(docs)
print("num of doc splits ",len(doc_splits))
# Print the first 5 chunks
for doc in doc_splits[:5]:
    print(doc.page_content)
    print("---------------------------------------------------")
