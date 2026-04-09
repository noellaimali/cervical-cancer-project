import sqlite3
import hashlib

DB_NAME = 'cytoscan.db'

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email=None, phone=None):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash, email, phone) VALUES (?, ?, ?, ?)", 
                       (username, hash_password(password), email, phone))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, role, phone FROM users WHERE username = ? AND password_hash = ?", 
                   (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user

def save_prediction(user_id, image_name, result, confidence, patient_name="", patient_age=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (user_id, image_name, result, confidence, patient_name, patient_age) VALUES (?, ?, ?, ?, ?, ?)", 
                   (user_id, image_name, result, confidence, patient_name, patient_age))
    conn.commit()
    conn.close()

def get_user_history(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT image_name, result, confidence, timestamp, patient_name, patient_age FROM predictions WHERE user_id = ? ORDER BY timestamp DESC", 
                   (user_id,))
    history = cursor.fetchall()
    conn.close()
    return history

def update_user_profile(user_id, email, phone=None, username=None):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        if username:
            cursor.execute("UPDATE users SET email = ?, phone = ?, username = ? WHERE id = ?", (email, phone, username, user_id))
        else:
            cursor.execute("UPDATE users SET email = ?, phone = ? WHERE id = ?", (email, phone, user_id))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def get_all_users():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, role, phone, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()
    return users

def get_all_predictions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.id, u.username, p.image_name, p.result, p.confidence, p.timestamp, p.patient_name, p.patient_age 
        FROM predictions p 
        LEFT JOIN users u ON p.user_id = u.id 
        ORDER BY p.timestamp DESC
    ''')
    preds = cursor.fetchall()
    conn.close()
    return preds

def get_patient_report(user_id=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    query = '''
        SELECT patient_name, patient_age, image_name, result, confidence, timestamp 
        FROM predictions 
        WHERE patient_name IS NOT NULL AND patient_name != ""
    '''
    if user_id:
        query += " AND user_id = ?"
        query += " ORDER BY patient_name ASC, timestamp DESC"
        cursor.execute(query, (user_id,))
    else:
        query += " ORDER BY patient_name ASC, timestamp DESC"
        cursor.execute(query)
    report = cursor.fetchall()
    conn.close()
    return report

def get_unique_patients(user_id=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    query = 'SELECT DISTINCT patient_name FROM predictions WHERE patient_name IS NOT NULL AND patient_name != ""'
    if user_id:
        query += " AND user_id = ?"
        query += " ORDER BY patient_name ASC"
        cursor.execute(query, (user_id,))
    else:
        query += " ORDER BY patient_name ASC"
        cursor.execute(query)
    patients = [row[0] for row in cursor.fetchall()]
    conn.close()
    return patients
