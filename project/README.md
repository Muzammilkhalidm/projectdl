# CyberAttend v2 — Face Recognition Attendance (Admin Panel + Employee Panel)

Do bilkul alag panels:

| Panel | URL | Login |
|---|---|---|
| **Admin** | `/admin/login` | username + password |
| **Employee** | `/login` | username + password **+ face recognition** |

## Employee ka poora flow

1. `/register` — naam, department, username, password.
2. **Face registration** — camera automatically 50 samples leta hai. Har sample `dataset/<id>/` mein save hota hai
   aur saath hi **`dataset_archive/dataset.zip`** mein pack hota rehta hai.
3. **"Admin's approval pending"** screen — page khud update hota hai.
4. Admin **Approvals** page par face samples dekh kar *Approve* karta hai.
5. Employee `/login` → password → **face scan** → Dashboard (face match na ho to login nahi hota).
6. Dashboard par **Check In (face scan)** → working timer chalta hai.
7. Kaam khatam karke **Check Out (face scan)** → working hours save. Agar required hours poore nahi hue to
   warning aati hai aur din **"Short Hours"** record hota hai (admin ko nazar aata hai).

## Department-wise access
Employee sirf **apni department** ki information dekh sakta hai (team members, department notices, attendance).
Doosri department ka URL kholne par `403` aata hai. IT Software ka employee sirf IT Software dekhta hai; baqi sab kuch sirf admin dekhta hai.

## Admin panel
Dashboard (aaj ki attendance + working hours, absent list, department chart) · Approvals (face samples ke saath) ·
Employees (approve / revoke / delete) · Attendance reports (date, department, employee, status filter + **CSV**) ·
Departments · Announcements (sab ko ya sirf ek department ko) · Settings (required hours, shift time, late grace, password) ·
**Dataset ZIP** download.

## Chalane ka tareeqa
```bash
python -m venv venv
venv\Scripts\activate            # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Browser: `http://127.0.0.1:5000` (employee) · `http://127.0.0.1:5000/admin/login` (admin)

Default admin: **muzammil / muzammil182921** — pehli dafa login ke baad **Settings → Change password** zaroor karein.

> **Camera:** browser camera sirf **HTTPS** ya **localhost** par chalta hai. Phone se LAN par test karna ho to
> `pip install cryptography` karke `HTTPS=1 python app.py` chalayen (self-signed certificate, browser warning accept karein).
> Railway par HTTPS pehle se hota hai.

## Settings / Environment variables
| Variable | Matlab | Default |
|---|---|---|
| `DATA_DIR` | database + dataset + trainer ka folder (Railway Volume path) | project folder |
| `SECRET_KEY` | session key (na ho to `secret.key` file auto-ban jati hai) | auto |
| `APP_TIMEZONE` | attendance ka time zone | `Asia/Karachi` |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | pehli dafa admin banane ke liye | muzammil / muzammil182921 |
| `CONFIDENCE_THRESHOLD` | face match strictness (chota = strict) | `75` |
| `STREAK_NEEDED` | login/check-in ke liye lagatar kitne frames match hon | `3` |
| `SAMPLES_NEEDED` | registration par face samples | `50` |
| `SYNC_KEY` | `sync_dataset.py` ke liye | — |
| `COOKIE_SECURE=1` | HTTPS par secure cookies | off |

**Railway:** Volume mount karein aur `DATA_DIR` us path par set karein, warna redeploy par database/dataset ud jayega.
`Procfile` gunicorn ko **1 worker** par chalata hai (zaroori hai).

## Files
```
app.py            routes (admin, employee, face verify)       database.py   SQLite + attendance/hours logic
face_utils.py     detection, training, 1:1 verification, zip  templates/     public pages, admin/, employee/
static/           css + js/face.js (camera)                   dataset/      face samples (dataset/<id>/n.jpg)
dataset_archive/  dataset.zip (auto)                          trainer/      trainer.yml (LBPH model)
```

## Zaroori limitations (imandari se)
* Recognition **OpenCV LBPH** hai — chhoti company ke liye theek, lekin deep-learning jitni mazboot nahi. Roshni acchi rakhein.
* **Liveness detection nahi hai:** koi employee ki *photo/video* camera ke saamne rakh de to fool ho sakta hai.
  Is liye admin approval aur face + password dono rakhe gaye hain. Sakht security chahiye to ArcFace/MediaPipe + blink check add karein.
* Ek din mein **ek check-in aur ek check-out**; raat 12 baje ke baad wali shift support nahi.
* Agar koi check-out bhool jaye to wo din admin ko **"No Check-out"** dikhta hai.
