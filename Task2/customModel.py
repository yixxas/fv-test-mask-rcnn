import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import random



def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def getResults(img):
	img = transforms.ToTensor()(img)
	model = get_instance_segmentation_model(2)
	model.load_state_dict(torch.load("./models/PennResnet50.pt",map_location='cpu'))
	model.eval()
	device = torch.device('cpu')
	with torch.no_grad():
		prediction = model([img.to(device)])
	noOfMasks = sum(prediction[0]['scores']>0.8).item()
	imgSize = img.size()[1] * img.size()[2]
	to_pil = transforms.ToPILImage()
	IMAGE = to_pil(img)	
	img_cv = cv2.cvtColor(np.array(IMAGE), cv2.COLOR_RGB2BGR)

	pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].cpu().detach().numpy())]
	masks = (prediction[0]['masks']>0.5).squeeze().detach().cpu().numpy()


	for i in range(noOfMasks):
		rgb_mask = random_colour_masks(masks[i])
		img_cv = cv2.addWeighted(img_cv,1,rgb_mask,0.5,0)
		cv2.rectangle(img_cv,pred_boxes[i][0],pred_boxes[i][1],color=(0,0,255),thickness = 2)
	cv2.imwrite("static/test.png",img_cv)
	return computeArea(prediction,noOfMasks,imgSize),img_cv

	

def computeArea(prediction,noOfMasks,imgSize):
	first = prediction[0]['masks'][0][0]
	for x in range(1,noOfMasks):
		new = prediction[0]['masks'][x][0]
		first = first.add(new)
	return (first.nonzero(as_tuple=False).size()[0]) / (imgSize)	
	

def random_colour_masks(image):

	colours = [[0, 0, 255],[0, 255, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
	r = np.zeros_like(image).astype(np.uint8)
	g = np.zeros_like(image).astype(np.uint8)
	b = np.zeros_like(image).astype(np.uint8)
	r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
	coloured_mask = np.stack([r, g, b], axis=2)
	return coloured_mask
