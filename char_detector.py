import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from function import *
from utils import merge_file

class CharDetector:
	def __init__(self):
		self.net_h = 416 
		self.net_w = 416
		self.anchors = [25,38, 34,47, 35,38, 42,44, 42,54, 49,50, 55,57, 63,64, 75,74]
		self.label = ['char']
		self.model = None
		self.obj_thresh = 0.5 
		self.nms_thresh = 0.3

	def load_model(self):
		merge_file('models/char_detector.h5', 6)
		self.model = load_model('models/char_detector.h5')

	def predict(self, image):
		numpy_image=np.array(image) 
		preprocess_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
		boxes = get_yolo_boxes(self.model, [preprocess_image], self.net_h, self.net_w, 
							self.anchors, self.obj_thresh, self.nms_thresh)[0]
		draw_boxes(preprocess_image, boxes, self.label, self.obj_thresh) 
		list_boxes_plate = []
		for box in boxes:
			if box.score != -1:
				list_boxes_plate.append((box.xmin, box.ymin, box.xmax, box.ymax))
		return list_boxes_plate