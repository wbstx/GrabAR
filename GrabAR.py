import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.autograd import Variable

import sys
sys.path.insert(0, "network")

img_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def opencv_to_pil(opencv_img):
	return Image.fromarray(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))


def pil_to_opencv(pil_img, channel=3):
	opencv_image = np.array(pil_img)
	if channel == 3:
		# Convert RGB to BGR
		opencv_image = opencv_image[:, :, ::-1].copy()

	return opencv_image


if __name__ == '__main__':
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	from handnet_mask import HandNetInitial

	net = HandNetInitial().to(device)
	net.load_state_dict(torch.load('44.pth.tar')['state_dict'])
	net.eval()

	from handnet_s import HandNet

	handseg_net = HandNet().to(device)
	handseg_net.load_state_dict(torch.load('hand_seg.tar')['state_dict'])

	hand_frame = cv2.imread('Image/hand.png')

	##################
	# You need to your AR system to generate these two images
	##################
	object_frame = cv2.imread('Image/object.png')
	object_mask_frame = cv2.imread('Image/object_mask.png')

	hand_frame = cv2.resize(hand_frame, (320, 320))
	object_frame = cv2.resize(object_frame, (320, 320))
	object_mask_frame = cv2.resize(object_mask_frame, (320, 320))

	pil_hand_frame = opencv_to_pil(hand_frame)
	pil_object_frame = opencv_to_pil(object_mask_frame)

	hand_var = Variable(img_transform(pil_hand_frame).unsqueeze(0)).to(device)
	object_var = Variable(img_transform(pil_object_frame).unsqueeze(0)).to(device)

	##################
	# Hand Segmetation Network
	##################
	with torch.no_grad():
		res = handseg_net(hand_var)
	confidence = res[0].data.squeeze(0).cpu().numpy()

	hand_mask = np.argmax(confidence, axis=0)
	hand_mask = np.uint8(hand_mask)
	hand_mask[np.where(hand_mask == 1)] = 255

	hand_frame[np.where(hand_mask != 255)] = np.array((119, 178, 78)).astype(np.uint8)
	cv2.imwrite('Image/hand_segmentation.png', hand_frame)

	pil_hand_frame = opencv_to_pil(hand_frame)
	hand_var = Variable(img_transform(pil_hand_frame).unsqueeze(0)).to(device)

	##################
	# Occlusion Estimation Network
	##################
	with torch.no_grad():
		res, _ = net(hand_var, object_var, object_var, torch.tensor([1]))
	confidence = res[0].data.squeeze(0).cpu().numpy()

	## mask ##
	mask = np.argmax(confidence, axis=0)
	mask = np.uint8(mask)
	mask[np.where(mask == 1)] = 128  # object
	mask[np.where(mask == 2)] = 255  # hand

	mask = cv2.medianBlur(mask, 7)
	ret, mask = cv2.threshold(mask, 129, 255, cv2.THRESH_BINARY)
	kernel = np.ones((3, 3), np.uint8)
	mask = cv2.erode(mask, kernel, iterations=1)

	##################
	# Generating final results based on the predicted mask
	##################
	object_frame[np.where(np.logical_and(mask==255, hand_mask==255))] = \
		hand_frame[np.where(np.logical_and(mask==255, hand_mask==255))]
	cv2.imwrite('Image/final_result.png', object_frame)