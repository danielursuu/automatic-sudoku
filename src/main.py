import cv2
import numpy as np

def SUDOKU_IMAGE_PATH():
    return "resources/sudoku_images/test_1.jpg"

def import_image_by_path(image_path):
    img = cv2.imread(image_path)

    return img

def get_thresholded_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray.copy(), (5,5), 0)
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

if __name__ == '__main__':
    main()