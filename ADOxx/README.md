
# Object-Detection-X-ADOxx

Object detection workstation implemented with ADOxx.
This project includes the features implemented following the [Background Image Model Type GraphRep](https://www.adoxx.org/live/faq/-/message_boards/message/87111) tutorial

## Requirements

Ensure to have pip, pillow, requests, and python installed.

> [!NOTE]
> To make the *Scene_Recognition* button works, please include the *send_image.py* script into the BOC folder (ADOxx installation folder, typicially is something like C:\Program Files\BOC\ADOxx_1.8.0)

## Object detection - Reading .json file containing image with objects detected by the algorithm
This button exectues the *Object_detection.asc* script: if no empty model is selected, this creates a new empty model and insert objects following the content of the .json file that is uploaded.

## Scene recognition - Sending image to be detected by the algorithm
This button exectues the *Scene_Recognition.asc* script: if no background image is selected, this permits to select a new background image which will be sent to the object detection algorithm.


### Software Compliance
This project is based on the last version of ADOxx (1.8.0).
