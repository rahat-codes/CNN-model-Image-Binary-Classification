import cv2, sys, time, pathlib

# usage: python collect.py train/mask
target = pathlib.Path("data") / sys.argv[1]
target.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
count = len(list(target.glob("*.jpg")))
print("SPACE = save photo, Q = quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.putText(frame, f"{target.name}: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("collect", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        cv2.imwrite(str(target / f"{int(time.time()*1000)}.jpg"), frame)
        count += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()