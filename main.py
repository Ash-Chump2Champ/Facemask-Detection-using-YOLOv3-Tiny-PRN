# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:10:47 2021

@author: Lenovo
"""

import pygame
import argparse
import cv2 
from supports import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.5, help='Confidence for yolov3 detection')
args = ap.parse_args()

classes = ['with_mask', 'improper_mask','no_mask']
#yolo = YOLO("models/mask-yolov3-tiny-prn.cfg", "models/mask-yolov3-tiny-prn.weights", classes)
yolo = YOLO("models/mask-yolov3-tiny-prn.cfg", "models/mask-yolov3-tiny-prn.weights", classes)

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

""" Initialize and Loading warning sound file"""
pygame.init()
pygame.mixer.set_num_channels(2)
voice = pygame.mixer.Channel(1)
warningsound = pygame.mixer.Sound('assets/warning.mp3')

""" Colors for classes bounding box
green: good
orange: bad
red: none
"""
colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]

print("starting webcam...")
cv2.namedWindow("LIVE")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    stats, frame = vc.read()

else:
    stats = False
    
while stats:
    width, height, inference_time, results = yolo.inference(frame)
    font = cv2.FONT_HERSHEY_PLAIN
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w/2)
        cy = y + (h/2)
        
        
        # draw a bounding box rectangle and label on the image
        color = colors[id]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        if (name == 'improper_mask'):
            frame = cv2.putText(frame, 'Warning: Incorrect Facemask Detected', (w//10, h//10), font, 1, (0, 0, 255), 2)
            if not voice.get_busy():
                voice.play(warningsound)
        elif (name == 'no_mask'):
            frame = cv2.putText(frame, 'Warning: No Facemask Detected',(w//10, h//10), font, 1, (0, 0, 255), 2)
            if not voice.get_busy():  # if the channel is not busy then play the warning sound
                voice.play(warningsound)
        else:
            frame = cv2.putText(frame, 'All good',(w//10, h//10), font, 1, (0, 0, 255), 2)
    cv2.imshow("LIVE", frame)
    
    stats, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("LIVE")
vc.release()