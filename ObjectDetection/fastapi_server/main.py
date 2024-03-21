from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn

from controller import image_processing


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/get-image/{uuid}", response_class=FileResponse)
async def get_image_predicted(uuid: str):
    image_path = "./predictions/" + uuid + ".jpg"
    return FileResponse(image_path, media_type="image/jpeg")


@app.post("/uploadfile/")
async def create_upload_file(image: UploadFile = File(...)):
    predictions =  await image_processing(image)
    return {"predictions": predictions}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
