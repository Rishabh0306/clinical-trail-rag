import ast
import os
import uuid
import time
import pandas as pd
import torch
from transformers import AutoTokenizer
from huggingface_hub import login
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RelevancyEvaluator

from config import *

# Read token from environment variable
token = os.getenv('HF_TOKEN')
login(token)


class ClinicalTrialRAG:
    def __init__(self):
        self.md_nodes = None
        self.index = None
        self.query_engine = None
        self.llm = None

    def load_data(self):
        documents = SimpleDirectoryReader(data_path).load_data()
        parser = SimpleFileNodeParser()
        self.md_nodes = parser.get_nodes_from_documents(documents)
        # Assign ids to keep the node ids consistent across runs
        namespace = uuid.NAMESPACE_DNS
        for idx, node in enumerate(self.md_nodes):
            name = f"node_{idx}"
            node.id_ = str(uuid.uuid5(namespace, name))

    def initialize_embeddings(self):
        embed_model = FastEmbedEmbedding(model_name=embedding_model_name, cache_dir=model_path)
        Settings.embed_model = embed_model
        Settings.chunk_size = 512

    def initialize_llm(self):
        system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
        # This will wrap the default prompts that are internal to llama-index
        query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=model_path)
        stopping_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>"), ]

        self.llm = HuggingFaceLLM(
            context_window=8192,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.7, "do_sample": False},
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=tokenizer_name,
            model_name=llm_model_name,
            device_map="cuda",
            stopping_ids=stopping_ids,
            tokenizer_kwargs={"max_length": 4096},
            model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True, "cache_dir": model_path}
        )

        Settings.llm = self.llm
        Settings.chunk_size = 512

    def initialize_vector_store(self):
        self.index = VectorStoreIndex(nodes=self.md_nodes)

    def initialize_query_engine(self):
        rerank = SentenceTransformerRerank(model=reranker_model_name, top_n=3)
        self.query_engine = self.index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank])

        now = time.time()
        response = self.query_engine.query("Which company has conducted trail for BIBF 1120?")
        print(f"Response Generated: {response}")
        print(f"Elapsed: {round(time.time() - now, 2)}s")

    def initialize_rag_pipeline(self):
        self.load_data()
        self.initialize_embeddings()
        self.initialize_llm()
        self.initialize_vector_store()
        self.initialize_query_engine()

    def generate_response(self, input_query):
        return self.query_engine.query(input_query)

    def generate_qa_dataset(self):
        if bl_generate_qa_dataset and not os.path.exists(qa_dataset_path):
            qa_dataset = generate_question_context_pairs(
                            self.md_nodes,
                            llm=self.llm,
                            num_questions_per_chunk=2
                        )

            qa_dataset.save_json(qa_dataset_path)

    def evaluate_rag(self):
        output_df = pd.DataFrame()

        test_data = pd.read_csv(test_dataset_path)

        # Retrieval Evaluation
        retriever = self.index.as_retriever(similarity_top_k=10)

        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )

        for idx, row in test_data.iterrows():
            eval_results = retriever_evaluator.evaluate(row['query'], expected_ids=ast.literal_eval(row['expected_ids']))
            metric_dict = eval_results.metric_vals_dict
            row['hit_rate'] = metric_dict['hit_rate']
            row['mrr'] = metric_dict['mrr']
            output_df = output_df._append(row, ignore_index=True)

        full_output_df = pd.DataFrame()

        # Response Evaluation
        for idx, row in output_df.iterrows():
            response_vector = self.query_engine.query(row['query'])
            row['response'] = str(response_vector)

            relevancy_evaluator = RelevancyEvaluator(llm=self.llm)

            relevancy_result = relevancy_evaluator.evaluate_response(
                query=row['query'], response=response_vector
            )

            row['relevancy_result_contexts'] = relevancy_result.contexts

            row['relevancy_result_passing'] = relevancy_result.passing

            row['relevancy_result_score'] = relevancy_result.score

            faithfullness_evaluator = FaithfulnessEvaluator(llm=self.llm)

            faithfullness_result = faithfullness_evaluator.evaluate_response(
                query=row['query'], response=response_vector
            )

            row['faithfullness_result_contexts'] = faithfullness_result.contexts

            row['faithfullness_result_passing'] = faithfullness_result.passing

            row['faithfullness_result_score'] = faithfullness_result.score

            full_output_df = full_output_df._append(row, ignore_index=True)

        full_output_df.to_csv(results_path, index=False)


if __name__ == '__main__':
    import nest_asyncio
    nest_asyncio.apply()
    ct_rag = ClinicalTrialRAG()
    ct_rag.initialize_rag_pipeline()
    ct_rag.evaluate_rag()





