from PIL import Image
import numpy as np
import io

from obj_detection_model.detectron_model import inference


async def image_processing(image):
    contents = await image.read()
    image = Image.open(io.BytesIO(contents))
    # image = _get_image_orientation(image)
    numpy_array = np.array(image)
    return inference(numpy_array)
