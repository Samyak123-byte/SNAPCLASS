from typing import List, Dict, Any

import bcrypt
from src.database.config import supabase


def hash_pass(pwd: str) -> str:
    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()


def check_pass(pwd: str, hashed: str) -> bool:
    return bcrypt.checkpw(pwd.encode(), hashed.encode())


def check_teacher_exists(username: str) -> bool:
    # Check for unique username, returns true when username is already taken
    response = (
        supabase.table("teachers")
        .select("username")
        .eq("username", username)
        .execute()
    )
    return len(response.data) > 0


def create_teacher(username, password, name):
    data = {
        "username": username,
        "password": hash_pass(password),
        "name": name,
    }
    response = supabase.table("teachers").insert(data).execute()
    return response.data


def teacher_login(username, password):
    response = (
        supabase.table("teachers").select("*").eq("username", username).execute()
    )
    if response.data:
        teacher = response.data[0]
        if check_pass(password, teacher["password"]):
            return teacher
    return None


def get_all_students():
    response = supabase.table("students").select("*").execute()
    return response.data


def create_student(new_name, face_embedding=None, voice_embedding=None):
    data = {
        "name": new_name,
        "face_embedding": face_embedding,
        "voice_embedding": voice_embedding,
    }
    response = supabase.table("students").insert(data).execute()
    return response.data


def create_subject(subject_code, name, section, teacher_id):
    data = {
        "subject_code": subject_code,
        "name": name,
        "section": section,
        "teacher_id": teacher_id,
    }
    response = supabase.table("subjects").insert(data).execute()
    return response.data


def get_teacher_subjects(teacher_id: int) -> List[Dict[str, Any]]:
    # Fixed Indentation Error
    response = (
        supabase.table("subjects")
        .select("*, subject_student(count), attendance_logs(timestamp)")
        .eq("teacher_id", teacher_id)
        .execute()
    )

    # Simplified Type Hinting (Removed broken JSON reference)
    subjects: List[Dict[str, Any]] = response.data

    for sub in subjects:
        # Fixed: Changed lookup key from 'subject_students' to 'subject_student'
        # Fixed: Supabase returns a direct dictionary for aggregate counts, not a list
        student_data = sub.get("subject_student")
        if isinstance(student_data, list) and len(student_data) > 0:
            sub["total_students"] = student_data[0].get("count", 0)
        elif isinstance(student_data, dict):
            sub["total_students"] = student_data.get("count", 0)
        else:
            sub["total_students"] = 0

        attendance = sub.get("attendance_logs", [])
        unique_sessions = len(set(log["timestamp"] for log in attendance))
        sub["total_classes"] = unique_sessions

        # Cleanup metadata keys
        sub.pop("subject_student", None)
        sub.pop("attendance_logs", None)

    return subjects


def enroll_student_to_subject(student_id, subject_id):
    data = {"student_id": student_id, "subject_id": subject_id}
    # Note: Use your singular table name if 'subject_students' causes an error
    response = supabase.table("subject_student").insert(data).execute()
    return response.data


def unenroll_student_to_subject(student_id, subject_id):
    # Note: Use your singular table name if 'subject_students' causes an error
    response = (
        supabase.table("subject_students")
        .delete()
        .eq("student_id", student_id)
        .eq("subject_id", subject_id)
        .execute()
    )
    return response.data


def get_student_subjects(student_id):
    # Note: Use your singular table name if 'subject_students' causes an error
    response = (
        supabase.table("subject_student")
        .select("*, subjects(*)")
        .eq("student_id", student_id)
        .execute()
    )
    return response.data


def get_student_attendance(student_id):
    response = (
        supabase.table("attendance_logs")
        .select("*, subjects(*)")
        .eq("student_id", student_id)
        .execute()
    )
    return response.data


def create_attendance(logs):
    response = supabase.table("attendance_logs").insert(logs).execute()
    return response.data


def get_attendance_for_teacher(teacher_id):
    response = (
        supabase.table("attendance_logs")
        .select("*, subjects!inner(*)")
        .eq("subjects.teacher_id", teacher_id)
        .execute()
    )
    return response.data


def mark_voice_attendance_db(student_id, subject_id, status=True):
    """
    Mark attendance for a student via voice verification
    
    Args:
        student_id: The student's unique ID
        subject_id: The subject's unique ID
        status: Attendance status (True = present, False = absent)
    """
    import sqlite3
    from datetime import datetime
    
    # Connect to database (update with your actual DB path)
    conn = sqlite3.connect('database.db')  # Change to your DB path
    cursor = conn.cursor()
    
    # Insert attendance record
    cursor.execute(
        '''INSERT INTO attendance (student_id, subject_id, status, attendance_type, timestamp) 
           VALUES (?, ?, ?, 'voice', ?)''',
        (student_id, subject_id, status, datetime.now())
    )
    
    conn.commit()
    conn.close()
    
    return True