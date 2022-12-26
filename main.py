import io
import os
import requests
from fastapi import FastAPI

from PIL import Image
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from imageai.Detection import ObjectDetection


detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("retinanet_resnet50_fpn_coco-eeacb38b.pth")
detector.loadModel()

app = FastAPI()

@app.get("/")
async def root():
    return "Hello World!!!"


@app.post("/object_detection")
async def obj_detect(image_url:str):
    
    response = requests.get(image_url)
    image_bytes = io.BytesIO(response.content)    
    original_image = Image.open(image_bytes)  #.convert("RGBA")

    original_image.save("input.jpg")
    img_path = "input.jpg"

    # detector = ObjectDetection()
    # detector.setModelTypeAsRetinaNet()
    # detector.setModelPath("resnet50.h5")
    # detector.loadModel()

    detection = detector.detectObjectsFromImage(input_image=img_path, output_image_path="result.jpg")
    # im = Image.open("result.jpg")
    # im.show()

    data = detection
    LIST = data
    shorted = ( ", ".join( repr(e) for e in LIST ))
    dic2 = eval(shorted)

    s_no = 1
    bb = [0]
    name = []

    if len(data) > 1 :

        for i in range (len(dic2)):
            Dict = dict(dic2[i])
            # print((Dict['name']))
            name.append((Dict['name']))
            # bb.append(Dict['box_points'])
            # s_no+=1

    else:
        Dict = dict(dic2)
        name.append((Dict['name']))

    print(name)
    return (name)