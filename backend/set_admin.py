import sqlite3
import sys

DB_PATH = "violations.db"

def set_admin():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Check if users table exists
    try:
        cur.execute("SELECT id, name, email, role FROM users")
        users = cur.fetchall()
    except sqlite3.OperationalError:
        print("Error: 'users' table not found. Did you run the API server at least once?")
        return

    if not users:
        print("No users found in the database.")
        print("Please register a user first via the website.")
        return

    print("\nExisting Users:")
    print(f"{'ID':<5} {'Name':<20} {'Email':<30} {'Current Role'}")
    print("-" * 70)
    for u in users:
        print(f"{u[0]:<5} {u[1]:<20} {u[2]:<30} {u[3]}")

    print("\n")
    target_email = input("Enter the Email of the user to promote to ADMIN: ").strip()

    if not target_email:
        print("Cancelled.")
        return

    cur.execute("UPDATE users SET role = 'admin' WHERE email = ?", (target_email,))
    
    if cur.rowcount > 0:
        conn.commit()
        print(f"\nSUCCESS! User '{target_email}' has been promoted to ADMIN.")
        print("Please log out and log back in for the changes to take effect.")
    else:
        print(f"\nError: User with email '{target_email}' not found.")

    conn.close()

if __name__ == "__main__":
    set_admin()
