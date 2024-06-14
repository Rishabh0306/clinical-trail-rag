from llama_index.core import SimpleDirectoryReader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import PromptTemplate

documents = SimpleDirectoryReader("/home/azureuser/clinical-trail-rag/data").load_data()


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings

vector_index = VectorStoreIndex.from_documents(documents)

query_engine = vector_index.as_query_engine()

ds_dict = {"question": "Which company has conducted trail for BIBF 1120?",
           "ground_truth": "Boehringer Ingelheim"}

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize LLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

stopping_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>"), ]

system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

llm = HuggingFaceLLM(
                    context_window=8192,
                    max_new_tokens=256,
                    generate_kwargs={"temperature": 0.7, "do_sample":False},
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
                    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                    device_map="cuda",
                    stopping_ids=stopping_ids,
                    tokenizer_kwargs={"max_length": 4096},
                    # uncomment this if using CUDA to reduce memory usage
                    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
                    )


from ragas.integrations.llama_index import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=ds_dict,
    llm=llm,
    embeddings=embed_model,
)

print(result)

df = result.to_pandas()

df.to_csv("/home/azureuser/clinical-trail-rag/results.csv", index=False)


