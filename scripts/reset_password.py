"""Set a new password for an existing account (use when someone is locked out).

Run inside the web container:
    docker compose exec app python scripts/reset_password.py user@example.com
Prints a new random password (or pass your own as the 2nd argument, 8+ characters).
"""
import os
import secrets
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo import MongoClient  # noqa: E402

import auth  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    email = sys.argv[1].strip().lower()
    pw = sys.argv[2] if len(sys.argv) > 2 else secrets.token_urlsafe(9)
    if len(pw) < 8:
        sys.exit("Password must be at least 8 characters.")
    db = MongoClient(os.environ["MONGO_URI"])[os.getenv("MONGO_DB", "transcriptor")]
    res = db.users.update_one({"email": email}, {"$set": {"password": auth.hash_password(pw)}})
    if not res.matched_count:
        sys.exit(f"No account with email {email}.")
    print(f"Password for {email} is now: {pw}")


if __name__ == "__main__":
    main()
