import cv2
# import time

cap = cv2.VideoCapture(0)  # 0表示默认摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

frame_count = 0
# st = time.time()
while True:
    ret, frame = cap.read()

    if not ret:
        print("无法读取视频帧")
        break

    frame_count += 1

    if frame_count % 2 == 0:  # 每隔一帧保存图像
        cv2.imwrite(f"extract/frame_{frame_count}.jpg", frame)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # ed = time.time()
    # print(ed - st)
    # if ed - st >= 1:  # 1s后停止视频抽帧
    #     break

cap.release()
cv2.destroyAllWindows()
