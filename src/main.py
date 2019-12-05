import cv2

def SUDOKU_IMAGE_PATH():
    return "resources/sudoku_images/test_1.jpg"

def import_image_by_path(image_path):
    img = cv2.imread(image_path)

    return img

def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey()



def main():
    original_image = import_image_by_path(SUDOKU_IMAGE_PATH())
    show_image("Original", original_image)

if __name__ == '__main__':
    main()