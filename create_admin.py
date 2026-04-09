import sqlite3
import hashlib
import argparse

DB_NAME = 'cytoscan.db'

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_admin(username, password, email=None, phone=None):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash, email, phone, role) VALUES (?, ?, ?, ?, ?)", 
                       (username, hash_password(password), email, phone, 'admin'))
        conn.commit()
        conn.close()
        print(f"[SUCCESS] Admin user '{username}' created successfully!")
        return True
    except sqlite3.IntegrityError:
        print(f"[ERROR] Username '{username}' already exists.")
        return False
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return False

def make_existing_user_admin(username):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET role = 'admin' WHERE username = ?", (username,))
        if cursor.rowcount > 0:
            print(f"[SUCCESS] User '{username}' is now an admin!")
        else:
            print(f"[ERROR] User '{username}' not found.")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage admin users for CytoScan.")
    parser.add_argument("action", choices=["create", "promote"], help="Action to perform: 'create' a new admin or 'promote' an existing user to admin.")
    parser.add_argument("username", help="Username")
    parser.add_argument("--password", help="Password (required for create)", default="")
    parser.add_argument("--email", help="Email", default=None)
    parser.add_argument("--phone", help="Phone", default=None)
    args = parser.parse_args()
    
    if args.action == "create":
        if not args.password:
            print("[ERROR] Password is required to create a new admin.")
        else:
            create_admin(args.username, args.password, args.email, args.phone)
    elif args.action == "promote":
        make_existing_user_admin(args.username)
