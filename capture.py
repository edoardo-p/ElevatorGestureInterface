import uuid

import cv2


def take_picture(path: str) -> None:
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()

        cv2.imshow("Hand recognizer", frame)
        key = chr(cv2.waitKey(1) & 0xFF)

        if key == " ":
            break
        elif key in ("0", "1", "2", "3", "4", "5", "u", "d", "o", "c"):
            cv2.imwrite(f"{path}/{key}/{uuid.uuid4()}", frame)

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    take_picture(r".\data")
