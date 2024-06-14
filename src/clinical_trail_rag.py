import os
import time
import asyncio
import pandas as pd
import torch
from transformers import AutoTokenizer
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
import qdrant_client
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.evaluation import generate_question_context_pairs, EmbeddingQAFinetuneDataset
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.evaluation import BatchEvalRunner

from config import *


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

    def initialize_embeddings(self):
        embed_model = FastEmbedEmbedding(model_name=embedding_model_name)
        Settings.embed_model = embed_model
        Settings.chunk_size = 512

    def initialize_llm(self):
        system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
        # This will wrap the default prompts that are internal to llama-index
        query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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
            model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
        )

        Settings.llm = self.llm
        Settings.chunk_size = 512

    def initialize_vector_store(self):
        client = qdrant_client.QdrantClient(location=":memory:")
        vector_store = QdrantVectorStore(client=client, collection_name="test")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex(nodes=self.md_nodes, storage_context=storage_context,)

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

    async def evaluate_rag(self):
        self.generate_qa_dataset()

        # Retrieval Evaluation
        qa_dataset = EmbeddingQAFinetuneDataset.from_json(qa_dataset_path)

        # retriever = self.index.as_retriever(similarity_top_k=10)
        #
        # retriever_evaluator = RetrieverEvaluator.from_metric_names(
        #     ["mrr", "hit_rate"], retriever=retriever
        # )
        #
        # eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
        #
        # def extract_results(name, eval_results):
        #     """Extract results from evaluate."""
        #
        #     metric_dicts = []
        #     for eval_result in eval_results:
        #         metric_dict = eval_result.metric_vals_dict
        #         metric_dicts.append(metric_dict)
        #
        #     full_df = pd.DataFrame(metric_dicts)
        #
        #     hit_rate = full_df["hit_rate"].mean()
        #     mrr = full_df["mrr"].mean()
        #
        #     metric_df = pd.DataFrame(
        #         {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
        #     )
        #
        #     return metric_df
        #
        # retrieval_results_df = extract_results("Retriever Results", eval_results)
        # retrieval_results_df.to_csv(retrieval_results_path, index=False)

        # Response Evaluation
        queries = list(qa_dataset.queries.values())

        queries = queries[267]

        response_vector = self.query_engine.query(queries)

        l = RelevancyEvaluator(llm=self.llm)

        eval_result = l.evaluate_response(
            query=queries, response=response_vector
        )

        print(eval_result)

        # runner = BatchEvalRunner(
        #     {"faithfulness": FaithfulnessEvaluator(), "relevancy": RelevancyEvaluator()},
        #         workers=1)
        #
        # eval_results = await runner.aevaluate_queries(
        #     self.query_engine, queries=queries
        # )
        #
        # faithfulness_score = sum(result.passing for result in eval_results['faithfulness']) / len(
        #     eval_results['faithfulness'])
        #
        # print(faithfulness_score)
        #
        # relevancy_score = sum(result.passing for result in eval_results['relevancy']) / len(eval_results['relevancy'])
        #
        # print(relevancy_score)

if __name__ == '__main__':
    ct_rag = ClinicalTrialRAG()
    ct_rag.initialize_rag_pipeline()
    asyncio.run(ct_rag.evaluate_rag())





