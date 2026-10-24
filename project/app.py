"""
app.py  -  Face Recognition Attendance System

Two completely separate panels:

  ADMIN PANEL     /admin/...   (username + password)
      approvals, employees, departments, attendance reports, announcements, settings, dataset zip

  EMPLOYEE PANEL  /login ... /dashboard
      1. Register (name, department, username, password)
      2. Face enrolment (samples are saved into dataset/ and packed into dataset.zip)
      3. "Waiting for admin approval" screen
      4. Admin approves
      5. Login = password + face recognition   ->  Dashboard
      6. Check-in / Check-out (each needs face recognition) -> working hours tracked
      7. An employee can only see their OWN department's information
"""
import csv
import io
import os
import secrets
import threading
import time
from collections import defaultdict
from datetime import timedelta
from functools import wraps

import cv2
from flask import (Flask, Response, abort, flash, g, jsonify, redirect, render_template,
                   request, send_file, session, url_for)

import database as db
import face_utils as fu

# ------------------------------------------------------------------ app setup
app = Flask(__name__)


def _load_secret_key():
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    path = os.path.join(db.DATA_DIR, "secret.key")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(secrets.token_hex(32))
    with open(path) as fh:
        return fh.read().strip()


app.secret_key = _load_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    MAX_CONTENT_LENGTH=4 * 1024 * 1024,
)

db.init_db()
if not os.path.exists(fu.ZIP_PATH):
    fu.rebuild_dataset_zip(db.list_employees())

MAX_FACE_FAILS = 25            # mismatching frames before the attempt is locked
PRE_AUTH_SECONDS = 10 * 60     # time allowed between password step and face step
_enroll_lock = threading.Lock()


# ------------------------------------------------------------------ security helpers
def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def csrf_protect():
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        sent = request.headers.get("X-CSRFToken") or request.form.get("csrf_token", "")
        if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
            abort(400, "Security token missing or expired. Please go back, refresh the page and try again.")


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(self)")
    if request.endpoint not in ("static",):
        resp.headers["Cache-Control"] = "no-store"
    return resp


_fails = defaultdict(list)


def throttled(key, limit=5, window=600):
    t = time.time()
    _fails[key] = [x for x in _fails[key] if t - x < window]
    return len(_fails[key]) >= limit


def register_fail(key):
    _fails[key].append(time.time())


def clear_fails(key):
    _fails.pop(key, None)


# ------------------------------------------------------------------ jinja helpers
@app.template_filter("hm")
def hm_filter(seconds):
    return db.fmt_hm(seconds)


@app.template_filter("dt")
def dt_filter(value):
    if not value:
        return "—"
    try:
        return db._parse(value).strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return value


@app.context_processor
def inject_globals():
    ctx = {"me": g.get("emp"), "admin": g.get("admin"), "SAMPLES_NEEDED": fu.SAMPLES_NEEDED}
    if g.get("admin"):
        ctx["pending_count"] = db.count_status("pending")
    return ctx


# ------------------------------------------------------------------ auth decorators
def admin_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if session.get("role") != "admin":
            return redirect(url_for("admin_login"))
        g.admin = db.get_admin(session.get("admin_id"))
        if not g.admin:
            session.clear()
            return redirect(url_for("admin_login"))
        return f(*a, **k)
    return wrapper


def employee_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if session.get("role") != "employee":
            return redirect(url_for("login"))
        emp = db.get_employee(session.get("employee_id"))
        if not emp or emp["status"] != "approved":
            session.clear()
            flash("Your access is no longer active. Please contact the admin.", "error")
            return redirect(url_for("login"))
        g.emp = emp
        return f(*a, **k)
    return wrapper


def pending_employee():
    """Employee who registered/logged in but is not yet fully approved."""
    emp_id = session.get("pending_id")
    return db.get_employee(emp_id) if emp_id else None


def pre_auth_employee():
    """Employee who passed the password step and now needs the face step."""
    emp_id = session.get("pre_auth_id")
    if not emp_id or time.time() - session.get("pre_auth_at", 0) > PRE_AUTH_SECONDS:
        return None
    emp = db.get_employee(emp_id)
    return emp if emp and emp["status"] == "approved" else None


def dataset_manifest():
    return db.list_employees()


# =================================================================== PUBLIC / EMPLOYEE AUTH
@app.route("/")
def index():
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    if session.get("role") == "employee":
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    departments = db.list_departments()
    form = {}
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip() for k in
                ("full_name", "username", "email", "designation", "department_id")}
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        err = None
        if not form["full_name"] or not form["username"] or not form["department_id"]:
            err = "Name, username and department are required."
        elif len(form["username"]) < 3 or not form["username"].replace("_", "").replace(".", "").isalnum():
            err = "Username must be 3+ characters (letters, numbers, . or _)."
        elif len(password) < 6:
            err = "Password must be at least 6 characters."
        elif password != confirm:
            err = "Passwords do not match."
        elif not db.get_department(int(form["department_id"] or 0)):
            err = "Please select a valid department."
        elif db.username_taken(form["username"]):
            err = "This username is already taken."
        if err:
            flash(err, "error")
        else:
            emp_id = db.create_employee(form["full_name"], form["username"], password,
                                        int(form["department_id"]), form["email"], form["designation"])
            session.clear()
            session["pending_id"] = emp_id
            flash("Account created. Now register your face.", "success")
            return redirect(url_for("enroll"))
    return render_template("register.html", departments=departments, form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("role") == "employee":
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        key = f"emp:{request.remote_addr}:{username.lower()}"
        if throttled(key):
            flash("Too many failed attempts. Please wait 10 minutes and try again.", "error")
            return render_template("login.html"), 429
        emp = db.get_employee_by_username(username)
        if not emp or not db.verify_password(emp, password):
            register_fail(key)
            flash("Invalid username or password.", "error")
            return render_template("login.html")
        clear_fails(key)
        session.clear()
        status = emp["status"]
        if status == "face_pending":
            session["pending_id"] = emp["id"]
            flash("Please finish registering your face.", "success")
            return redirect(url_for("enroll"))
        if status == "pending":
            session["pending_id"] = emp["id"]
            return redirect(url_for("waiting"))
        if status == "rejected":
            flash("Your account was rejected or its access was revoked by the admin.", "error")
            return render_template("login.html")
        session["pre_auth_id"] = emp["id"]
        session["pre_auth_at"] = time.time()
        return redirect(url_for("login_face"))
    return render_template("login.html")


@app.route("/login/face")
def login_face():
    emp = pre_auth_employee()
    if not emp:
        flash("Please enter your username and password first.", "error")
        return redirect(url_for("login"))
    return render_template("face_scan.html", layout="public_base.html", purpose="login",
                           title="Face Verification", subject=emp, early=False, remaining=0)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ------------------------------------------------------------------ enrolment + waiting
@app.route("/enroll")
def enroll():
    emp = pending_employee()
    if not emp:
        return redirect(url_for("login"))
    if emp["status"] == "pending":
        return redirect(url_for("waiting"))
    if emp["status"] != "face_pending":
        return redirect(url_for("login"))
    fu.reset_samples(emp["id"])          # always start a clean capture
    db.set_samples(emp["id"], 0)
    return render_template("enroll.html", emp=emp, needed=fu.SAMPLES_NEEDED)


@app.route("/api/enroll/frame", methods=["POST"])
def api_enroll_frame():
    emp = pending_employee()
    if not emp or emp["status"] != "face_pending":
        return jsonify(ok=False, error="Session expired. Please login again."), 403
    frame = fu.decode_image((request.get_json(silent=True) or {}).get("image", ""))
    if frame is None:
        return jsonify(ok=False, error="Invalid image."), 400
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = fu.detect_faces(gray)
    count = fu.count_samples(emp["id"])
    if len(faces) == 0:
        return jsonify(ok=True, state="no_face", count=count)
    x, y, w, h = [int(v) for v in fu.largest_face(faces)]
    if any(f[2] * f[3] > 0.5 * w * h and tuple(int(v) for v in f) != (x, y, w, h) for f in faces):
        return jsonify(ok=True, state="multiple_faces", count=count)

    with _enroll_lock:
        count = fu.count_samples(emp["id"])
        if count >= fu.SAMPLES_NEEDED:
            return jsonify(ok=True, state="done", count=count, done=True, redirect=url_for("waiting"))
        count = fu.save_face_sample(emp["id"], gray[y:y + h, x:x + w])
        if count < fu.SAMPLES_NEEDED:
            return jsonify(ok=True, state="captured", count=count, box=[x, y, w, h],
                           frame=[frame.shape[1], frame.shape[0]])
        # ---- enrolment complete ----
        similar = fu.find_similar_existing(emp["id"])
        note = ""
        if similar:
            other = db.get_employee(similar[0])
            if other:
                note = (f"Face looks {int(similar[1] * 100)}% similar to existing employee "
                        f"'{other['full_name']}' (#{other['id']}). Please verify identity before approving.")
        ok, msg = fu.train_model()
        if not ok:
            return jsonify(ok=False, error=msg), 500
        db.set_samples(emp["id"], count)
        db.set_status(emp["id"], "pending", flag_note=note)
        fu.rebuild_dataset_zip(dataset_manifest())
    return jsonify(ok=True, state="done", count=count, done=True, redirect=url_for("waiting"))


@app.route("/waiting")
def waiting():
    emp = pending_employee()
    if not emp:
        return redirect(url_for("login"))
    if emp["status"] == "face_pending":
        return redirect(url_for("enroll"))
    return render_template("waiting.html", emp=emp)


@app.route("/api/status")
def api_status():
    emp = pending_employee()
    if not emp:
        return jsonify(status="none"), 401
    return jsonify(status=emp["status"])


@app.route("/waiting/continue")
def waiting_continue():
    """After approval the employee must log in normally (password + face)."""
    session.clear()
    return redirect(url_for("login"))


# =================================================================== FACE VERIFY (login + check in/out)
@app.route("/api/face/verify", methods=["POST"])
def api_face_verify():
    data = request.get_json(silent=True) or {}
    purpose = data.get("purpose")
    if purpose == "login":
        emp = pre_auth_employee()
    elif purpose in ("checkin", "checkout") and session.get("role") == "employee":
        emp = db.get_employee(session.get("employee_id"))
        if emp and emp["status"] != "approved":
            emp = None
    else:
        return jsonify(ok=False, error="Bad request."), 400
    if not emp:
        return jsonify(ok=False, state="auth", error="Session expired. Please login again.",
                       redirect=url_for("login")), 401

    frame = fu.decode_image(data.get("image", ""))
    if frame is None:
        return jsonify(ok=False, error="Invalid image."), 400
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = fu.verify_face(gray, emp["id"])
    out = {"ok": True, "state": res["state"], "needed": fu.STREAK_NEEDED,
           "box": res.get("box"), "frame": [frame.shape[1], frame.shape[0]]}

    if res["state"] in ("no_face", "multiple_faces", "no_model"):
        out["streak"] = (session.get("face_state") or {}).get("streak", 0) if res["state"] != "multiple_faces" else 0
        return jsonify(out)

    st = session.get("face_state") or {}
    if st.get("purpose") != purpose or st.get("emp") != emp["id"]:
        st = {"purpose": purpose, "emp": emp["id"], "streak": 0, "fails": 0}
    if res["match"]:
        st["streak"] += 1
    else:
        st["streak"], st["fails"] = 0, st["fails"] + 1

    if st["fails"] >= MAX_FACE_FAILS:
        session.pop("face_state", None)
        if purpose == "login":
            session.pop("pre_auth_id", None)
            out.update(state="locked", error="Face did not match. Please login again.", redirect=url_for("login"))
        else:
            out.update(state="locked", error="Face did not match. Please try again.")
        return jsonify(out)

    out["streak"] = st["streak"]
    if st["streak"] < fu.STREAK_NEEDED:
        session["face_state"] = st
        return jsonify(out)

    # ----- verified: perform the action right here (no client-side flag to forge) -----
    session.pop("face_state", None)
    if purpose == "login":
        session.clear()
        session.permanent = True
        session["role"] = "employee"
        session["employee_id"] = emp["id"]
        out.update(state="verified", message=f"Welcome, {emp['full_name']}!", redirect=url_for("dashboard"))
    elif purpose == "checkin":
        ok, msg = db.check_in(emp["id"])
        flash(msg, "success" if ok else "error")
        out.update(state="verified" if ok else "blocked", message=msg, redirect=url_for("dashboard"))
    else:
        ok, msg = db.check_out(emp["id"], allow_early=bool(data.get("early_ok")))
        flash(msg, "success" if ok else "error")
        out.update(state="verified" if ok else "blocked", message=msg, redirect=url_for("dashboard"))
    return jsonify(out)


# =================================================================== EMPLOYEE PANEL
@app.route("/dashboard")
@employee_required
def dashboard():
    emp = g.emp
    settings = db.get_settings()
    req = db.required_seconds(settings)
    rec = db.get_today_record(emp["id"])
    today = db.enrich(rec, req) if rec else None
    return render_template(
        "employee/dashboard.html", active="dashboard", settings=settings, req=req, today=today,
        summary=db.month_summary(emp["id"]), history=db.employee_history(emp["id"], limit=7),
        announcements=db.list_announcements(emp["department_id"], limit=3),
        server_now=db.now().strftime(db.DT_FMT),
    )


@app.route("/attendance/scan/<action>")
@employee_required
def scan(action):
    if action not in ("checkin", "checkout"):
        abort(404)
    emp = g.emp
    rec = db.get_today_record(emp["id"])
    req = db.required_seconds()
    early, remaining = False, 0
    if action == "checkin" and rec:
        flash("You have already checked in today.", "error")
        return redirect(url_for("dashboard"))
    if action == "checkout":
        if not rec:
            flash("You have not checked in today.", "error")
            return redirect(url_for("dashboard"))
        if rec["check_out"]:
            flash("You have already checked out today.", "error")
            return redirect(url_for("dashboard"))
        e = db.enrich(rec, req)
        if e["worked"] < req:
            early, remaining = True, req - e["worked"]
    return render_template("face_scan.html", layout="employee/base.html", active="dashboard",
                           purpose=action, subject=emp, early=early, remaining=remaining,
                           title="Check In" if action == "checkin" else "Check Out")


@app.route("/my-attendance")
@employee_required
def my_attendance():
    month = request.args.get("month") or db.now().strftime("%Y-%m")
    return render_template("employee/attendance.html", active="attendance", month=month,
                           rows=db.employee_history(g.emp["id"], month),
                           summary=db.month_summary(g.emp["id"], month))


@app.route("/department")
@employee_required
def my_department():
    return redirect(url_for("department", dept_id=g.emp["department_id"]))


@app.route("/department/<int:dept_id>")
@employee_required
def department(dept_id):
    if dept_id != g.emp["department_id"]:
        abort(403)                       # employees only ever see THEIR department
    return render_template("employee/department.html", active="department",
                           dept=db.get_department(dept_id), members=db.department_members(dept_id),
                           announcements=db.list_announcements(dept_id, limit=20))


@app.route("/profile", methods=["GET", "POST"])
@employee_required
def profile():
    if request.method == "POST":
        cur, new, conf = (request.form.get(k, "") for k in ("current", "new", "confirm"))
        if not db.verify_password(g.emp, cur):
            flash("Current password is wrong.", "error")
        elif len(new) < 6:
            flash("New password must be at least 6 characters.", "error")
        elif new != conf:
            flash("New passwords do not match.", "error")
        else:
            db.set_employee_password(g.emp["id"], new)
            flash("Password updated.", "success")
        return redirect(url_for("profile"))
    return render_template("employee/profile.html", active="profile",
                           dept=db.get_department(g.emp["department_id"]))


# =================================================================== ADMIN PANEL
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        key = f"adm:{request.remote_addr}:{username.lower()}"
        if throttled(key):
            flash("Too many failed attempts. Please wait 10 minutes.", "error")
            return render_template("admin/login.html"), 429
        row = db.get_admin_by_username(username)
        if not db.verify_password(row, request.form.get("password", "")):
            register_fail(key)
            flash("Invalid username or password.", "error")
            return render_template("admin/login.html")
        clear_fails(key)
        session.clear()
        session.permanent = True
        session["role"] = "admin"
        session["admin_id"] = row["id"]
        return redirect(url_for("admin_dashboard"))
    return render_template("admin/login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    today_rows = db.attendance_report(date_from=db.today_str(), date_to=db.today_str())
    return render_template(
        "admin/dashboard.html", active="dashboard", stats=db.admin_stats(),
        pending=db.list_employees(status="pending"), today_rows=today_rows,
        absent=db.absent_today(), depts=db.departments_overview(),
        settings=db.get_settings(), req=db.required_seconds())


@app.route("/admin/approvals")
@admin_required
def admin_approvals():
    return render_template("admin/approvals.html", active="approvals",
                           pending=db.list_employees(status="pending"),
                           enrolling=db.list_employees(status="face_pending"))


def _admin_employee_action(emp_id, status, message):
    emp = db.get_employee(emp_id)
    if not emp:
        abort(404)
    if status == "approved" and emp["status"] not in ("pending", "rejected"):
        flash("This employee has not finished face registration yet.", "error")
    else:
        db.set_status(emp_id, status)
        flash(message.format(name=emp["full_name"]), "success")
    return redirect(request.referrer or url_for("admin_employees"))


@app.route("/admin/employee/<int:emp_id>/approve", methods=["POST"])
@admin_required
def admin_approve(emp_id):
    return _admin_employee_action(emp_id, "approved", "{name} approved. They can now login with face recognition.")


@app.route("/admin/employee/<int:emp_id>/reject", methods=["POST"])
@admin_required
def admin_reject(emp_id):
    return _admin_employee_action(emp_id, "rejected", "{name} rejected / access revoked.")


@app.route("/admin/employee/<int:emp_id>/delete", methods=["POST"])
@admin_required
def admin_delete_employee(emp_id):
    emp = db.get_employee(emp_id)
    if not emp:
        abort(404)
    db.delete_employee(emp_id)
    fu.delete_employee_data(emp_id, dataset_manifest())
    flash(f"{emp['full_name']} and their face data were deleted.", "success")
    return redirect(url_for("admin_employees"))


@app.route("/admin/employees")
@admin_required
def admin_employees():
    status = request.args.get("status") or None
    dept = request.args.get("department_id", type=int)
    q = request.args.get("q", "").strip() or None
    return render_template("admin/employees.html", active="employees",
                           employees=db.list_employees(status, dept, q),
                           departments=db.list_departments(), f_status=status, f_dept=dept, f_q=q or "")


@app.route("/admin/employee/<int:emp_id>")
@admin_required
def admin_employee_detail(emp_id):
    emp = db.get_employee(emp_id)
    if not emp:
        abort(404)
    month = request.args.get("month") or db.now().strftime("%Y-%m")
    n = fu.count_samples(emp_id)
    picks = sorted({i for i in (1, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n) if i >= 1}) if n else []
    return render_template("admin/employee_detail.html", active="employees", emp=emp, month=month,
                           rows=db.employee_history(emp_id, month), summary=db.month_summary(emp_id, month),
                           sample_ids=picks, sample_total=n)


@app.route("/admin/employee/<int:emp_id>/sample/<int:n>.jpg")
@admin_required
def admin_sample(emp_id, n):
    p = fu.sample_path(emp_id, n)
    if not p:
        abort(404)
    return send_file(p, mimetype="image/jpeg")


@app.route("/admin/departments", methods=["GET", "POST"])
@admin_required
def admin_departments():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Department name is required.", "error")
        elif db.add_department(name, request.form.get("description", "")):
            flash("Department added.", "success")
        else:
            flash("A department with this name already exists.", "error")
        return redirect(url_for("admin_departments"))
    return render_template("admin/departments.html", active="departments", depts=db.departments_overview())


@app.route("/admin/departments/<int:dept_id>/delete", methods=["POST"])
@admin_required
def admin_delete_department(dept_id):
    if db.delete_department(dept_id):
        flash("Department deleted.", "success")
    else:
        flash("Cannot delete a department that still has employees.", "error")
    return redirect(url_for("admin_departments"))


def _report_args():
    today = db.today_str()
    return dict(
        date_from=request.args.get("from") or today,
        date_to=request.args.get("to") or today,
        department_id=request.args.get("department_id", type=int),
        employee_id=request.args.get("employee_id", type=int),
        status=request.args.get("status") or None,
    )


@app.route("/admin/attendance")
@admin_required
def admin_attendance():
    args = _report_args()
    rows = db.attendance_report(**args)
    done = [r for r in rows if r["check_out"]]
    return render_template("admin/attendance.html", active="attendance", rows=rows, a=args,
                           departments=db.list_departments(),
                           employees=db.list_employees(status="approved"),
                           total_worked=sum(r["worked"] for r in done), req=db.required_seconds(),
                           qs=request.query_string.decode())


@app.route("/admin/attendance.csv")
@admin_required
def admin_attendance_csv():
    rows = db.attendance_report(**_report_args())
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Date", "Employee", "Username", "Department", "Check In", "Check Out",
                "Worked (hh:mm)", "Required (hh:mm)", "Status", "Late"])
    for r in rows:
        w.writerow([r["work_date"], r["full_name"], r["username"], r["department_name"],
                    r["check_in_time"], r["check_out_time"], db.fmt_hm(r["worked"]).replace("h ", ":").replace("m", ""),
                    db.fmt_hm(r["required"]).replace("h ", ":").replace("m", ""), r["status"],
                    "Yes" if r["is_late"] else "No"])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=attendance_report.csv"})


@app.route("/admin/announcements", methods=["GET", "POST"])
@admin_required
def admin_announcements():
    if request.method == "POST":
        title, body = request.form.get("title", "").strip(), request.form.get("body", "").strip()
        dept = request.form.get("department_id", type=int)
        if not title or not body:
            flash("Title and message are required.", "error")
        else:
            db.add_announcement(dept, title, body)
            flash("Announcement posted.", "success")
        return redirect(url_for("admin_announcements"))
    return render_template("admin/announcements.html", active="announcements",
                           items=db.list_announcements(everything=True, limit=100),
                           departments=db.list_departments())


@app.route("/admin/announcements/<int:ann_id>/delete", methods=["POST"])
@admin_required
def admin_delete_announcement(ann_id):
    db.delete_announcement(ann_id)
    flash("Announcement deleted.", "success")
    return redirect(url_for("admin_announcements"))


@app.route("/admin/settings", methods=["GET", "POST"])
@admin_required
def admin_settings():
    if request.method == "POST":
        if request.form.get("section") == "password":
            cur, new, conf = (request.form.get(k, "") for k in ("current", "new", "confirm"))
            if not db.verify_password(g.admin, cur):
                flash("Current password is wrong.", "error")
            elif len(new) < 8:
                flash("New admin password must be at least 8 characters.", "error")
            elif new != conf:
                flash("New passwords do not match.", "error")
            else:
                db.set_admin_password(g.admin["id"], new)
                flash("Admin password updated.", "success")
        else:
            try:
                hours = float(request.form.get("required_hours", ""))
                hh, mm = request.form.get("shift_start", "09:00").split(":")
                int(hh), int(mm)
                grace = int(request.form.get("late_grace_minutes", "15"))
                if not (0.5 <= hours <= 16) or grace < 0:
                    raise ValueError
                db.save_settings(hours, f"{int(hh):02d}:{int(mm):02d}", grace)
                flash("Working-hours settings saved.", "success")
            except ValueError:
                flash("Please enter valid values (hours between 0.5 and 16).", "error")
        return redirect(url_for("admin_settings"))
    return render_template("admin/settings.html", active="settings", s=db.get_settings())


@app.route("/admin/dataset.zip")
@admin_required
def admin_dataset_zip():
    fu.rebuild_dataset_zip(dataset_manifest())
    return send_file(fu.ZIP_PATH, mimetype="application/zip", as_attachment=True, download_name="dataset.zip")


@app.route("/api/dataset_zip")
def dataset_zip_sync():
    """Used by sync_dataset.py:  /api/dataset_zip?key=<SYNC_KEY env var>"""
    sync_key = os.environ.get("SYNC_KEY", "")
    provided = request.args.get("key", "")
    if not (session.get("role") == "admin" or (sync_key and secrets.compare_digest(provided, sync_key))):
        return jsonify(error="Unauthorized"), 401
    fu.rebuild_dataset_zip(dataset_manifest())
    return send_file(fu.ZIP_PATH, mimetype="application/zip", as_attachment=True, download_name="dataset.zip")


# ------------------------------------------------------------------ errors
@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
def http_error(e):
    return render_template("error.html", code=e.code, message=getattr(e, "description", "")), e.code


if __name__ == "__main__":
    ssl_ctx = "adhoc" if os.environ.get("HTTPS") == "1" else None   # needs: pip install cryptography
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True,
            debug=os.environ.get("DEBUG") == "1", ssl_context=ssl_ctx)
