from fastapi import FastAPI , Request 
import numpy as np
from fastai.vision.all import PILImage
import base64

app = FastAPI()

def read64(uri):
    encoded_data = uri.split(',')[1]
    img = PILImage.create(base64.b64decode(encoded_data))
    return img

@app.get("/api/gethog")
async def read_str(request : Request):
    item = await request.json()
    item_str = item['img']
    result=read64(item_str)
    return {"result":result}

