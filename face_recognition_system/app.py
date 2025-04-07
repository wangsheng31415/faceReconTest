from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import face_recognition
from face_utils import get_face_encodings
from database import save_face, load_faces

app = Flask(__name__)

# 人脸识别阈值
THRESHOLD = 0.6

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    # 接收前端传来的图片和名字
    name = request.form.get('name')
    image_file = request.files.get('image')
    
    # 将图片转为numpy数组
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # 获取人脸编码
    encodings = get_face_encodings(img)
    if len(encodings) == 0:
        return jsonify({'success': False, 'error': '未检测到人脸'})
    
    # 保存到数据库
    save_face(name, encodings[0])
    return jsonify({'success': True})

@app.route('/recognize', methods=['POST'])
def recognize():
    # 接收实时帧图片
    image_file = request.files.get('image')
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # 获取当前帧的人脸编码
    current_encodings = get_face_encodings(img)
    if len(current_encodings) == 0:
        return jsonify({'name': '未知'})
    
    # 从数据库加载已注册人脸
    registered_faces = load_faces()
    if not registered_faces:
        return jsonify({'name': '未知'})
    
    # 比对所有人脸
    registered_encodings = [face['encoding'] for face in registered_faces]
    distances = face_recognition.face_distance(registered_encodings, current_encodings[0])
    min_distance = np.min(distances)
    
    if min_distance < THRESHOLD:
        best_match_index = np.argmin(distances)
        return jsonify({'name': registered_faces[best_match_index]['name']})
    else:
        return jsonify({'name': '未知'})

if __name__ == '__main__':
    app.run(debug=True)