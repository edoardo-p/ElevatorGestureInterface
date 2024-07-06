import cv2


def take_picture(path: str) -> None:
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()

        cv2.imshow("Hand recognizer", frame)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            cv2.imwrite(path, frame)
            break

    video.release()
    cv2.destroyAllWindows()
