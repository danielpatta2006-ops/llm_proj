"""SQLite database layer: users, profiles, resumes, interviews, answers."""
import hashlib, hmac, json, os, sqlite3
from datetime import datetime
from .config import DB_PATH


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL, pw_hash TEXT NOT NULL, created_at TEXT);
        CREATE TABLE IF NOT EXISTS profiles(
            user_id INTEGER PRIMARY KEY, full_name TEXT, email TEXT, branch TEXT,
            college TEXT, grad_year INTEGER, target_role TEXT, level TEXT);
        CREATE TABLE IF NOT EXISTS resumes(
            user_id INTEGER PRIMARY KEY, raw_text TEXT, analysis_json TEXT, updated_at TEXT);
        CREATE TABLE IF NOT EXISTS interviews(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, role TEXT, category TEXT,
            difficulty TEXT, started_at TEXT, finished_at TEXT, overall_score REAL,
            report_json TEXT);
        CREATE TABLE IF NOT EXISTS answers(
            id INTEGER PRIMARY KEY AUTOINCREMENT, interview_id INTEGER, q_index INTEGER,
            question TEXT, qtype TEXT, topic TEXT, answer TEXT, score REAL, eval_json TEXT);
        """)


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000).hex()


def register(username, password):
    username = username.strip().lower()
    if len(username) < 3 or len(password) < 6:
        return False, "Username needs 3+ characters and password 6+ characters."
    salt = os.urandom(16).hex()
    try:
        with _conn() as c:
            c.execute("INSERT INTO users(username,salt,pw_hash,created_at) VALUES(?,?,?,?)",
                      (username, salt, _hash(password, salt), datetime.now().isoformat()))
        return True, "Account created. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def login(username, password):
    """Return user id or None."""
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username.strip().lower(),)).fetchone()
    if row and hmac.compare_digest(row["pw_hash"], _hash(password, row["salt"])):
        return row["id"]
    return None


def save_profile(uid, **f):
    with _conn() as c:
        c.execute("""INSERT INTO profiles(user_id,full_name,email,branch,college,grad_year,target_role,level)
            VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET full_name=excluded.full_name,
            email=excluded.email, branch=excluded.branch, college=excluded.college,
            grad_year=excluded.grad_year, target_role=excluded.target_role, level=excluded.level""",
                  (uid, f.get("full_name"), f.get("email"), f.get("branch"), f.get("college"),
                   f.get("grad_year"), f.get("target_role"), f.get("level")))


def get_profile(uid):
    with _conn() as c:
        r = c.execute("SELECT * FROM profiles WHERE user_id=?", (uid,)).fetchone()
    return dict(r) if r else {}


def save_resume(uid, text, analysis):
    with _conn() as c:
        c.execute("""INSERT INTO resumes(user_id,raw_text,analysis_json,updated_at) VALUES(?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET raw_text=excluded.raw_text,
            analysis_json=excluded.analysis_json, updated_at=excluded.updated_at""",
                  (uid, text, json.dumps(analysis), datetime.now().isoformat()))


def get_resume(uid):
    with _conn() as c:
        r = c.execute("SELECT * FROM resumes WHERE user_id=?", (uid,)).fetchone()
    if not r:
        return None
    return {"raw_text": r["raw_text"], "analysis": json.loads(r["analysis_json"])}


def create_interview(uid, role, category, difficulty):
    with _conn() as c:
        cur = c.execute("INSERT INTO interviews(user_id,role,category,difficulty,started_at) VALUES(?,?,?,?,?)",
                        (uid, role, category, difficulty, datetime.now().isoformat()))
        return cur.lastrowid


def save_answer(iid, idx, q, answer, evaluation):
    with _conn() as c:
        c.execute("""INSERT INTO answers(interview_id,q_index,question,qtype,topic,answer,score,eval_json)
                     VALUES(?,?,?,?,?,?,?,?)""",
                  (iid, idx, q["question"], q.get("type"), q.get("topic"), answer,
                   evaluation["overall"], json.dumps(evaluation)))


def finish_interview(iid, overall, report):
    with _conn() as c:
        c.execute("UPDATE interviews SET finished_at=?, overall_score=?, report_json=? WHERE id=?",
                  (datetime.now().isoformat(), overall, json.dumps(report), iid))


def list_interviews(uid):
    with _conn() as c:
        rows = c.execute("""SELECT id,role,category,difficulty,started_at,overall_score
            FROM interviews WHERE user_id=? AND finished_at IS NOT NULL ORDER BY id DESC""", (uid,)).fetchall()
    return [dict(r) for r in rows]


def get_interview(iid):
    with _conn() as c:
        i = c.execute("SELECT * FROM interviews WHERE id=?", (iid,)).fetchone()
        a = c.execute("SELECT * FROM answers WHERE interview_id=? ORDER BY q_index", (iid,)).fetchall()
    if not i:
        return None
    d = dict(i)
    d["report"] = json.loads(d["report_json"]) if d["report_json"] else {}
    d["answers"] = [dict(x, eval=json.loads(x["eval_json"])) for x in a]
    return d


def all_user_answers(uid):
    with _conn() as c:
        rows = c.execute("""SELECT a.topic, a.score, a.eval_json, i.started_at FROM answers a
            JOIN interviews i ON i.id=a.interview_id WHERE i.user_id=? AND i.finished_at IS NOT NULL""",
                         (uid,)).fetchall()
    return [dict(r) for r in rows]
