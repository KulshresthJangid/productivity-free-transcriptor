"""Password hashing + JWT session cookie. Users live in MongoDB."""
import os
import time

import bcrypt
import jwt
from bson import ObjectId
from fastapi import HTTPException, Request

SECRET = os.getenv("JWT_SECRET", "")
COOKIE = "session"
MAX_AGE = 60 * 60 * 24 * 7  # 7 days
SECURE_COOKIE = os.getenv("COOKIE_SECURE", "false").lower() == "true"  # true behind HTTPS


def check_secret():
    if len(SECRET) < 32:
        raise RuntimeError("JWT_SECRET must be set to a random string of 32+ characters (see .env.example).")


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode()[:72], bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode()[:72], hashed.encode())
    except ValueError:
        return False


def make_token(user_id: str) -> str:
    return jwt.encode({"sub": user_id, "exp": int(time.time()) + MAX_AGE}, SECRET, algorithm="HS256")


def set_session(response, user_id: str):
    response.set_cookie(COOKIE, make_token(user_id), max_age=MAX_AGE, httponly=True,
                        samesite="lax", secure=SECURE_COOKIE)


async def current_user(request: Request):
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(401, "Not logged in")
    try:
        uid = jwt.decode(token, SECRET, algorithms=["HS256"])["sub"]
        user = await request.app.state.db.users.find_one({"_id": ObjectId(uid)})
    except Exception:
        user = None
    if not user:
        raise HTTPException(401, "Session expired")
    return user
