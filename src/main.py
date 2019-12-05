import cv2

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

def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey()


def main():
    original_image = import_image_by_path(SUDOKU_IMAGE_PATH())
    show_image("Original", original_image)

    processed_image = get_thresholded_image(original_image)
    show_image("Treshold", processed_image)

if __name__ == '__main__':
    main()