import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.siliconcloud import siliconcloud_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
load_dotenv() #配置APIKEY,写在.env中

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        # "QPro/deepseek-ai/DeepSeek-R1", #满血版
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", #Distill版
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url="https://api.siliconflow.cn/v1/",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await siliconcloud_embedding(
        texts,
        # model="BAAI/bge-large-zh-v1.5",
        model="BAAI/bge-large-en-v1.5",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        # max_token_size=512,
    )

async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("LLMAITEST1--------------------How are you?")
    print("llm_model_func: ", result)
    
    result = await embedding_func(["EMBEDDINGTEST2----------------------How are you?"])
    print("embedding_func: ", result)


asyncio.run(test_funcs())
##################################################
async def main():
    try:
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=8192,
                func=embedding_func,
            ),
        )
        # filePath = "电力物联网零信任架构下的分布式认证模型_唐大圆.pdf"
        
        # Extract text from the PDF file
        # pdf_text = extract_pdf_text(filePath)
        
    #     if pdf_text:
    #         print(f"Text length: {len(pdf_text)}")
    #         # Insert the extracted text into the LightRAG model
    #         await rag.ainsert(pdf_text)

    #     # Perform naive search
    #     print(
    #         await rag.aquery(
    #             "What are the top themes in this paper?", param=QueryParam(mode="naive")
    #         )
    #     )

    #     # Perform local search
    #     print(
    #         await rag.aquery(
    #             "What are the top themes in this paper?", param=QueryParam(mode="local")
    #         )
    #     )

    #     # Perform global search
    #     print(
    #         await rag.aquery(
    #             "What are the top themes in this paper?",
    #             param=QueryParam(mode="global"),
    #         )
    #     )

    #     # Perform hybrid search
    #     print(
    #         await rag.aquery(
    #             "What are the top themes in this paper?",
    #             param=QueryParam(mode="hybrid"),
    #         )
    #     )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())

# rag = LightRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=llm_model_func,
#     embedding_func=EmbeddingFunc(
#         embedding_dim=768, 
#         max_token_size=512, 
#         func=embedding_func
#     ),
# )


# with open("./book.txt") as f:
#     rag.insert(f.read())

# # Perform naive search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# )

# # Perform local search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
# )

# # Perform global search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
# )

# # Perform hybrid search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
# )
