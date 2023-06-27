import cv2


def main():
    camera = cv2.VideoCapture(0)

    # while True:
    
    ret_val, img = camera.read()
    results = img
    cv2.imshow("img",img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()




