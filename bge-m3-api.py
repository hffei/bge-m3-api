from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel
import numpy as np

app = FastAPI()

# 加载模型
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

class EmbeddingRequest(BaseModel):
    input: list

class EmbeddingResponse(BaseModel):
    data: list

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    try:
        embeddings = model.encode(request.input, batch_size=12, max_length=8192)['dense_vecs']
        response_data = [{"object": "embedding", "embedding": emb.tolist(), "index": idx} for idx, emb in enumerate(embeddings)]
        return {"data": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7005)
