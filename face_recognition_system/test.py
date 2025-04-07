import cv2
import face_recognition
import numpy as np

# 步骤1：加载已知人脸数据（假设已知人脸图片存放在 `known_faces` 文件夹）
known_face_encodings = []
known_face_names = []

# 示例：加载一张已知人脸（替换为你的图片路径）
known_image = face_recognition.load_image_file("test.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
known_face_encodings.append(known_encoding)
known_face_names.append("wang sheng")

# 步骤2：打开摄像头
video_capture = cv2.VideoCapture(0)

while True:
    # 读取摄像头画面
    ret, frame = video_capture.read()
    if not ret:
        break

    # 将图像从BGR转换为RGB（face-recognition库需要RGB格式）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 检测人脸位置和编码
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # 遍历检测到的人脸
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 与已知人脸比对
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # 计算匹配度（取最小距离）--欧氏距离，一般小于0.6认为是同一个人
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # 绘制人脸框和标签
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Face Recognition", frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
video_capture.release()
cv2.destroyAllWindows()