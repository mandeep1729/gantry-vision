import os
import cv2
from ultralytics import YOLO
from datetime import datetime


PIXEL_TO_REAL = {}
MASK_X0, MASK_Y0 = (320, 60)
MASK_X1, MASK_Y1 = (1175, 875)

REAL_WIDTH = 32
REAL_HEIGHT = 17.5


model = YOLO("models/potatos.pt")

os.makedirs('data/history', exist_ok=True)


def lookup_basler(x, y, img_width, img_height):
    x_real = x * REAL_WIDTH/img_width
    y_real = y * REAL_HEIGHT/img_height
    x_coord = x_real - REAL_WIDTH/2
    y_coord = REAL_HEIGHT/2 - y_real

    return round(x_coord, 2), round(y_coord, 2)


def detect_potatos(image):
    """
    Detects the presence of a potato in the given image.

    Args:
        image (str): Path to the input image file.

    Returns:
        list: List of (X, Y) coordinates.
    """
    # roi = image[MASK_Y0:MASK_Y1, MASK_X0:MASK_X1]
    results = model(image, conf=0.5)

    detections = []
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    h, w, _ = image.shape
    if results and results[0].boxes:
        boxes = [b.xyxy.tolist()[0] for b in results[0].boxes]
        cv2.imwrite(f'data/history/{now}_{len(boxes)}.jpg', image)
        for x0, y0, x1, y1 in boxes:
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 200), 2)
            xc, yc = (x0 + x1)//2, (y0 + y1)//2
            px_xy = lookup_basler(xc, yc, w, h)
            detections += [px_xy]
            image = cv2.rectangle(image, (x0, y0), (x1, y1), (100, 0, 200), 2)
            image = cv2.putText(image, f'{px_xy}', (xc, yc), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
    # image = cv2.rectangle(image, (MASK_X0, MASK_Y0), (MASK_X1, MASK_Y1), (0, 50, 200), 1)
    cv2.imwrite(f'data/history/{now}_detected.jpg', image)
    return detections


if __name__ == '__main__':
    img = cv2.imread('data/sample.jpg')
    result = detect_potatos(img)
    print('Potato Coordinates for data/sample.jpg are:', result)
