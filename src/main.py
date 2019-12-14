import cv2
import numpy as np

def SUDOKU_IMAGE_PATH():
    return "resources/sudoku_images/test_1.jpg"

def import_image_by_path(image_path):
    img = cv2.imread(image_path)

    return img

def get_thresholded_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray.copy(), (5,5), 3)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    return thresh

def get_mask_with_max_area(contours):
    max_area = 0
    mask = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > max_area):
            max_area = area
            mask = contour

    return mask

def crop_grid(img):
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = get_mask_with_max_area(contours)

    bounding_rect = cv2.boundingRect(mask)
    x, y, w, h = bounding_rect
    oppening = np.zeros((h, w), np.float)
    oppening[0:h, 0:w] = img[y:y + h, x:x + w]

    return oppening, bounding_rect

def filter_image(oppening, bounding_rect):
    kernel = np.zeros((11,11),dtype=np.uint8)
    kernel[5,...] = 1
    line = cv2.morphologyEx(oppening,cv2.MORPH_OPEN, kernel, iterations=2)
    oppening-=line

    x, y, w, h = bounding_rect
    oppening_cann = np.empty((h, w), np.uint8)
    oppening_cann[:, :] = oppening[:, :]

    lines = cv2.HoughLinesP(oppening_cann, 1, np.pi/180, 106, 80, 10)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(oppening,(x1,y1),(x2,y2),(0,0,0), 3)

    oppening = cv2.morphologyEx(oppening,cv2.MORPH_OPEN, (2, 3), iterations=1)

    return oppening, bounding_rect


def infer_grid(img):
	"""Infers 81 cell grid from a square image."""
	squares = []
	side = img.shape[:1]
	side = side[0] / 9
	for i in range(9):
		for j in range(9):
			p1 = (i * side, j * side)  # Top left corner of a bounding box
			p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
			squares.append((p1, p2))

	return squares

def display_rects(in_img, rects, colour=255):
	"""Displays rectangles on the image."""
	img = in_img.copy()
	for rect in rects:
		img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
    
	return img


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey()


def main():
    original_image = import_image_by_path(SUDOKU_IMAGE_PATH())
    show_image("Original", original_image)

    processed_image = get_thresholded_image(original_image)
    show_image("Treshold", processed_image)

    crop, rectangle = crop_grid(processed_image)
    show_image("Cropped", crop)

    squares = infer_grid(crop)
    img = display_rects(crop,squares)
    show_image("Rectangles", img)
   

if __name__ == '__main__':
    main()