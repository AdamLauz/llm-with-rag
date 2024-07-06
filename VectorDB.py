import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage

# set number of docs to retrieve
TOP_K = 3

# import any embedding model on HF hub (https://huggingface.co/spaces/mteb/leaderboard)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large") # alternative model
Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25

QUERY_ENGINE_FILE = 'models/query_engine.pkl'
PERSIST_DIR = "./storage"


def build_index():
    documents = SimpleDirectoryReader("articles").load_data()
    print(f"Num of docs {len(documents)}")

    # Store docs into vector DB
    index = VectorStoreIndex.from_documents(documents)
    save_index(index)
    return index


def get_query_engine():
    index = load_index()
    if index is None:
        print("Index doesn't exist, building new index from the provided documents")
        index = build_index()

    # Configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )

    # Assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

    return query_engine


def save_index(index):
    index.storage_context.persist(persist_dir=PERSIST_DIR)


def load_index():
    if not os.path.exists(PERSIST_DIR):
        return None
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))


def get_context(query: str, query_engine):
    # query documents
    response = query_engine.query(query)

    # reformat response
    context = "Context:\n"
    for i in range(TOP_K):
        context = context + response.source_nodes[i].text + "\n\n"

    return context


if __name__ == "__main__":
    query_engine = get_query_engine()

    context = get_context("who are the Circassians?", query_engine)
    print(context)
