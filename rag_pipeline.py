import time
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.core.postprocessor import SentenceTransformerRerank


# Load data
documents = SimpleDirectoryReader("C:\\Users\\risha\Desktop\Projects\clinical-trail-rag\data").load_data()

# Initialize Embeddings
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.embed_model = embed_model

Settings.chunk_size = 512

# Define System Prompt

system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

# Initialize LLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

stopping_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>"), ]

llm = HuggingFaceLLM(
                    context_window=8192,
                    max_new_tokens=256,
                    generate_kwargs={"temperature": 0.7, "do_sample":False},
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
                    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                    device_map="cpu",
                    stopping_ids=stopping_ids,
                    tokenizer_kwargs={"max_length": 4096},
                    # uncomment this if using CUDA to reduce memory usage
                    # model_kwargs={"torch_dtype": torch.float16}
                    )

Settings.llm = llm
Settings.chunk_size = 512

# Initialize Vector Store
client = qdrant_client.QdrantClient(
# you can use :memory: mode for fast and light-weight experiments,
# it does not require to have Qdrant deployed anywhere
# but requires qdrant-client >= 1.1.1
location=":memory:"
# otherwise set Qdrant instance address with:
# url="http://<host>:<port>"
# otherwise set Qdrant instance with host and port:
#host="localhost",
#port=6333
# set API KEY for Qdrant Cloud
#api_key=<YOUR API KEY>
)

vector_store = QdrantVectorStore(client=client,collection_name="test")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context,)

# Initialize Reranker
rerank = SentenceTransformerRerank( model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)

# Initialize query engine
query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank])

now = time.time()
response = query_engine.query("What is the Background Information and Trial Rationale for Protocol Number DNDi-CH-E1224-003?",)
print(f"Response Generated: {response}")
print(f"Elapsed: {round(time.time() - now, 2)}s")



