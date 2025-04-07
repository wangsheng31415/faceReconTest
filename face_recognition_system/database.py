import sqlite3
import json
import numpy as np

def init_db():
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  encoding TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def save_face(name, encoding):
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    # 将numpy数组转为JSON字符串存储
    encoding_json = json.dumps(encoding.tolist())
    c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding_json))
    conn.commit()
    conn.close()

def load_faces():
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM faces")
    faces = []
    for row in c.fetchall():
        name, encoding_json = row
        encoding = np.array(json.loads(encoding_json))
        faces.append({'name': name, 'encoding': encoding})
    conn.close()
    return faces

# 初始化数据库
init_db()