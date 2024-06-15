from pathlib import Path

__WORKSPACE__ = Path(__file__).parent.parent

data_path = str(__WORKSPACE__ / "data")
embedding_model_name = "BAAI/bge-small-en-v1.5"
llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer_name = "meta-llama/Meta-Llama-3-8B-Instruct"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-2-v2"
bl_generate_qa_dataset = False
qa_dataset_path = str(__WORKSPACE__ / "test_data" / "synthetic_test_data.json")
test_dataset_path = str(__WORKSPACE__ / "test_data" / "test_data.csv")
results_path = str(__WORKSPACE__ / "test_data" / "results.csv")
model_path = str(__WORKSPACE__ / "models")