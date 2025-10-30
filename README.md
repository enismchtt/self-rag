#SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (ICLR 2024, Oral top 1%)
Self-RAG paper's implementation with **LangGraph** on **VespaDB**.  
Can be used for **local document RAG** setups.  
Uses **Ollama Gemma-3** model for lightweight local inference.

---

### 🚀 How to Run

#### 1. Adjust your local documents  
Edit `app.py` and set your local .pdf document paths in the `docs` section.

#### 2. Start Ollama  
```bash
ollama pull gemma  # or any model you like but you also have to edit it in the code
```
#### 3. Start VespaDB
```bash
python setup_vespa.py  # make sure Docker is running first
```
#### 4. Run the app with Chainlit
```bash
chainlit run app.py
```
