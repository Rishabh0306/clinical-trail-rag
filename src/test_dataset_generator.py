import os
import openai
import torch
from llama_index.core import SimpleDirectoryReader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM

# os.environ['OPENAI_API_KEY'] = ""
# openai.api_key = os.environ['OPENAI_API_KEY']

# Load data
documents = SimpleDirectoryReader("C:\\Users\\risha\Desktop\Projects\clinical-trail-rag\data").load_data()

# Initialize Embeddings
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
# # This will wrap the default prompts that are internal to llama-index
# query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")


# Initialize LLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

stopping_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>"), ]

llm = HuggingFaceLLM(
                    context_window=8192,
                    max_new_tokens=256,
                    generate_kwargs={"temperature": 0.7, "do_sample":False},
                    # system_prompt=system_prompt,
                    # query_wrapper_prompt=query_wrapper_prompt,
                    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
                    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                    device_map="cuda",
                    stopping_ids=stopping_ids,
                    tokenizer_kwargs={"max_length": 4096},
                    # uncomment this if using CUDA to reduce memory usage
                    model_kwargs={"torch_dtype": torch.float16}
                    )


# # generator with openai models
# generator_llm = OpenAI(model="gpt-3.5-turbo")
# critic_llm = OpenAI(model="gpt-3.5-turbo")
# embeddings = OpenAIEmbedding(model="text-embedding-3-small", api_key="")

generator_llm = llm
critic_llm = llm
embeddings = embed_model

generator = TestsetGenerator.from_llama_index(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings,
)

# generate testset
testset = generator.generate_with_llamaindex_docs(
    documents,
    test_size=20,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)

df = testset.to_pandas()
df.to_csv("C:\\Users\\risha\Desktop\Projects\clinical-trail-rag\\test_data.csv", index=False)

