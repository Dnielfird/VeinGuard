import sqlite3
from datetime import datetime
import os

DB_PATH = 'app/students.db'

def init_db():
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Student registration table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            name TEXT,
            matric TEXT PRIMARY KEY,
            email TEXT,
            department TEXT,
            phone TEXT,
            course_code TEXT,
            section TEXT,
            class_time TEXT
        )
    ''')

    # Attendance or enrollment logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            matric TEXT,
            timestamp TEXT,
            type TEXT
        )
    ''')

    conn.commit()
    conn.close()

def register_student(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO students (
            name, matric, email, department, phone, 
            course_code, section, class_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['name'], data['matric'], data['email'], data['department'],
        data['phone'], data['course_code'], data['section'], data['class_time']
    ))
    conn.commit()
    conn.close()

def get_student_info_by_matric(matric):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE matric = ?", (matric,))
    row = cursor.fetchone()
    conn.close()
    if row:
        keys = ['name', 'matric', 'email', 'department', 'phone', 'course_code', 'section', 'class_time']
        return dict(zip(keys, row))
    return {}

def log_activity(name, matric, log_type="verification"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO logs (name, matric, timestamp, type) 
        VALUES (?, ?, ?, ?)
    ''', (name, matric, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), log_type))
    conn.commit()
    conn.close()

def get_logs_by_matric(matric):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs WHERE matric = ?", (matric,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_all_students():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_all_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs")
    rows = cursor.fetchall()
    conn.close()
    return rows

def check_admin_login(username, password):
    return username == "admin" and password == "pass123"
