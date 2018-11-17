from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
import uvicorn, aiohttp, asyncio
import base64
import sys
import numpy as np
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path

path = Path(__file__).parent

model_file_url = 'https://github.com/pankymathur/Fine-Grained-Clothing-Classification/blob/master/data/cloth_categories/models/stage-1_sz-150.pth?raw=true'
model_file_name = 'model'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

MODEL_PATH = path/'models'/f'{model_file_name}.h5'
IMG_FILE_SRC = path/'static'/'saved_image.png'
PREDICTION_FILE_SRC = path/'static'/'predictions.txt'

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
    await download_file(model_file_url, MODEL_PATH)
    # Load your Custom trained model
    # model = load_model(MODEL_PATH)
    # model._make_predict_function()
    # print('Your Custom Trained Model loaded...')
    model = ResNet50(weights='imagenet')
    print('Default Imagenet Model loaded..')
    return model

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["img"].read())
    bytes = base64.b64decode(img_bytes)
    with open(IMG_FILE_SRC, 'wb') as f: f.write(bytes)
    return model_predict(IMG_FILE_SRC, model)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))
    preds = model.predict(x)
    # Get Top-3 Accuracy
    predictions = decode_predictions(preds, top=3)[0]
    result = []
    for p in predictions:
        _,label,accuracy = p
        result.append((label,accuracy))
    with open(PREDICTION_FILE_SRC, 'w') as f: f.write(str(result))
    result_html = path/'static'/'result.html'
    return HTMLResponse(result_html.open().read())

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=5042)