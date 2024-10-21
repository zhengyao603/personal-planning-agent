from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from component.utils import load_config

def get_chroma_db() -> Chroma:
    chroma_db = Chroma(
        collection_name="memo",
        embedding_function=OpenAIEmbeddings(model=load_config()["llm"]["embedding"]),
        persist_directory=load_config()["chroma"]["directory"],
    )
    return chroma_db