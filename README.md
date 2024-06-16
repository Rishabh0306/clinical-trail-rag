# clinical-trail-rag
 A RAG system capable of answering questions related to given clinical pdfs.

### System Configurations:

CPU(s): 4 

GPU: Nvidia Tesla T4 16GB

RAM: 28GB

Python Version: 3.10.12

### How to Run:

1. Clone the repository
2. Using docker:  
   1. docker build -t test_rag -f app.Dockerfile .
   2. docker run -d --gpus all --name my_test_rag_container -p 9090:9090 -e HF_TOKEN=<HUGGINGFACE TOKEN> -v <MODEL_DIRECTORY_PATH>:/clinical-trial-rag/models test_rag
3. Using Python:
   1. Setup virtual environment: python3 -m venv venv 
   2. Activate virtual environment: source venv/bin/activate
   3. Install requirements: pip install -r requirements.txt 
   4. Run RAG app: python3 src/app.py
   5. Run evaluation: python3 src/clinical_trail_rag.py

### Benchmarks:
1. [CPU Usage](benchmarks/cpu_usage.png)
2. [Memory Usage](benchmarks/memory_usage.png)
3. [GPU Usage](benchmarks/gpu_usage.png)
4. [GPU Memory Usage](benchmarks/gpu_memory_usage.png)

### Evaluation Metrics:

| S no.     | Embedding Model        | Reranker Model                       | LLM Model                           | Hit Rate | MRR  | Faithfulness | Answer Relevancy |
|-----------|------------------------|--------------------------------------|-------------------------------------|----------|------|--------------|------------------|
| Feature 1 | BAAI/bge-small-en-v1.5 | cross-encoder/ms-marco-MiniLM-L-2-v2 | meta-llama/Meta-Llama-3-8B-Instruct | 0.53     | 0.38 | 0.53         | 1                |
| Feature 2 | BAAI/bge-small-en-v1.5 | cross-encoder/ms-marco-MiniLM-L-2-v2 | aaditya/OpenBioLLM-Llama3-8B        | 0.53     | 0.38 | 0.33         | 0.13             |

