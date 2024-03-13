import requests
import sys

def send_image(image_path):
    try:
        # server URL 
        url = 'http://localhost:3456/upload'

        # read the file in binary mode
        with open(image_path, 'rb') as file:
            files = {'image': file}
            response = requests.post(url, files=files)

            # Controls request's status
            if response.status_code == 200:
                print("Image sent successfully!")
            else:
                print("An error occured during the image sending. Status code:", response.status_code)
    except Exception as e:
        print("An error occured during the image sending:", str(e))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python send_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    send_image(image_path)
