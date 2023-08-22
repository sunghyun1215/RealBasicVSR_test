import cv2


def main():
    video = cv2.VideoCapture(1)

    video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    print("video start")
    
    while True:
    
        ret_val, img = video.read()

        cv2.imshow("img",img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()




