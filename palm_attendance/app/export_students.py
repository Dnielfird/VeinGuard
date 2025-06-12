import sqlite3
import csv

DB_PATH = '/home/aapuzi/palm_attendance/app/students.db'
CSV_PATH = '/home/aapuzi/palm_attendance/app/exported_students.csv'

def export_students_to_csv():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students")
    rows = cursor.fetchall()

    headers = ['Name', 'Matric No', 'Email', 'Department', 'Phone', 'Course Code', 'Section', 'Class Time']

    with open(CSV_PATH, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    conn.close()
    print(f"Exported {len(rows)} students to {CSV_PATH}")

if __name__ == "__main__":
    export_students_to_csv()
