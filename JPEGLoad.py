import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# image_path = r'idd-detection\IDD_Detection\JPEGImages\frontFar\BLR-2018-03-22_17-39-26_2_frontFar\000006_r.jpg'
# image = load_image(image_path)
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()