import sys
import os
from PIL import Image

def resize_image(image_path, width_desired):
    try:
        # open the image
        image = Image.open(image_path)

        width, height = image.size

        # if the image that has been passed already has the width that is desired, skip
        if width == width_desired:
            print("Image already has the desired width.")
            return

        # proportions caluclation
        proportions = height / width

        # image resizing
        resized_image = image.resize((width_desired, round(width_desired * proportions)))

        # get resized image path
        file_name, file_extension = os.path.splitext(image_path)
        resized_image_path = f"{file_name}_resized{file_extension}"

        # save resized image
        resized_image.save(resized_image_path)
        
        print("Image has been resized and saved at:", resized_image_path)
    except Exception as e:
        print("Something wrong during the resizing of the image:", str(e))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python resize_image.py <image_path>.png <width_desired>")
    else:
        image_path = sys.argv[1]
        width_desired = int(sys.argv[2])
        resize_image(image_path, width_desired)