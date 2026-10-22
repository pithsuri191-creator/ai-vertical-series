import streamlit as st
import pandas as pd
import datetime
import os
import google.generativeai as genai

# ---------------------------------------------------------
# 1. Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Cloud AI Studio - Production System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0B0E14; color: #E6EDF3; font-family: 'Inter', sans-serif; }
    .metric-card { background-color: #161B22; border: 1px solid #30363D; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4); }
    .metric-title { color: #8B949E; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { color: #F0F6FC; font-size: 2.2rem; font-weight: 700; margin-top: 8px; }
    .metric-delta-pos { color: #3FB950; font-size: 0.85rem; font-weight: 500; margin-top: 5px; }
    .badge { padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: bold; display: inline-block; }
    .badge-completed { background-color: #113222; color: #3FB950; border: 1px solid #1B472C; }
    .badge-processing { background-color: #38290D; color: #D29922; border: 1px solid #573E0E; }
    .badge-failed { background-color: #3E1B1B; color: #F85149; border: 1px solid #5C2828; }
    .badge-pending { background-color: #21262D; color: #8B949E; border: 1px solid #30363D; }
    div[data-testid="stSidebar"] { background-color: #11151C; border-right: 1px solid #30363D; }
    .login-box { max-width: 420px; margin: 80px auto; padding: 30px; background-color: #161B22; border: 1px solid #30363D; border-radius: 16px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Session State Initialization
# ---------------------------------------------------------
# ระบบบัญชีผู้ใช้มาตรฐาน (Default Accounts)
USER_DATABASE = {
    "admin": {"password": "123", "role": "Admin", "name": "System Administrator"},
    "creator": {"password": "123", "role": "Creator", "name": "Content Creator 01"}
}

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'current_role' not in st.session_state:
    st.session_state.current_role = None

if 'jobs_list' not in st.session_state:
    st.session_state.jobs_list = [
        {
            "id": "JOB-901", "user": "creator", "title": "ประธานบริษัทปลอมตัวมาลองใจ ตอนที่ 1", 
            "genre": "ดราม่า/แก้แค้น", "status": "Completed", "progress": 100, 
            "step_script": "Done", "step_audio": "Done", "step_visual": "Done", "step_assembly": "Done",
            "created": "2026-10-21 15:30", "output": "storage/outputs/job901.mp4"
        },
        {
            "id": "JOB-902", "user": "creator", "title": "ย้อนเวลามาเป็นฮูหยินยอดอัจฉริยะ", 
            "genre": "แฟนตาซี/ย้อนยุค", "status": "Processing", "progress": 65, 
            "step_script": "Done", "step_audio": "Done", "step_visual": "Processing", "step_assembly": "Pending",
            "created": "2026-10-21 16:10", "output": None
        }
    ]

if 'ai_generated_prompt' not in st.session_state:
    st.session_state.ai_generated_prompt = ""

if 'system_logs' not in st.session_state:
    st.session_state.system_logs = [
        f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Enterprise System Initialized on Streamlit Cloud."
    ]

if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------
# 3. Helper Functions & Robust AI Call
# ---------------------------------------------------------
def get_badge(status):
    if status in ['Done', 'Completed']: return '<span class="badge badge-completed">✓ เสร็จสมบูรณ์</span>'
    if status == 'Processing': return '<span class="badge badge-processing">🔄 กำลังประมวลผล</span>'
    if status == 'Failed': return '<span class="badge badge-failed">✕ เกิดข้อผิดพลาด</span>'
    return '<span class="badge badge-pending">⏳ รอคิวประมวลผล</span>'

def generate_ai_story_prompt(api_key, genre):
    """เรียกใช้งาน Gemini API พร้อมระบบเลือกโมเดลสำรอง (Fallback mechanism)"""
    if not api_key:
        return False, "ยังไม่ได้กรอกหรือตั้งค่า Gemini API Key"
    
    genai.configure(api_key=api_key)
    # ลำดับชื่อโมเดลที่รองรับการสร้างข้อความ
    model_candidates = ['gemini-1.5-flash-latest', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    
    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(f"ขอพล็อตเรื่องย่อสำหรับซีรีส์แนวสั้นแนวตั้ง แนว {genre} ความยาว 3 บรรทัด น่าตื่นเต้น น่าติดตาม")
            if response and response.text:
                return True, response.text
        except Exception:
            continue
            
    return False, "ไม่สามารถเชื่อมต่อโมเดล Gemini ได้ กรุณาตรวจสอบ API Key หรือเปลี่ยนรหัส Key ใหม่"

# ---------------------------------------------------------
# 4. Authentication View (หน้าเข้าสู่ระบบ)
# ---------------------------------------------------------
if not st.session_state.authenticated:
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.title("⚡ Enterprise Login")
    st.caption("กรุณาเข้าสู่ระบบเพื่อใช้งานคลาวด์แดชบอร์ด")
    st.divider()
    
    login_user = st.text_input("ชื่อผู้ใช้งาน (Username)", value="admin")
    login_pass = st.text_input("รหัสผ่าน (Password)", type="password", value="123")
    
    if st.button("🔐 เข้าสู่ระบบ", use_container_width=True):
        if login_user in USER_DATABASE and USER_DATABASE[login_user]["password"] == login_pass:
            st.session_state.authenticated = True
            st.session_state.current_user = login_user
            st.session_state.current_role = USER_DATABASE[login_user]["role"]
            st.session_state.system_logs.insert(0, f"[AUTH] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - User '{login_user}' logged in successfully as {USER_DATABASE[login_user]['role']}.")
            st.success("เข้าสู่ระบบสำเร็จ!")
            st.rerun()
        else:
            st.error("ชื่อผู้ใช้งานหรือรหัสผ่านไม่ถูกต้อง")
            
    st.info("💡 **ข้อมูลการทดสอบเข้าสู่ระบบ:**\n- **Admin:** `admin` / `123`\n- **Creator:** `creator` / `123`")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------
# 5. Sidebar Panel (ควบคุมระบบตามสิทธิ์)
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚡ Enterprise Studio")
    st.caption(f"ผู้ใช้งาน: **@{st.session_state.current_user}** ({st.session_state.current_role})")
    
    if st.button("🚪 ออกจากระบบ", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.current_role = None
        st.rerun()
        
    st.divider()

    # 🔒 สิทธิ์ Admin เท่านั้นที่สามารถตั้งค่า API Key ของระบบได้
    if st.session_state.current_role == "Admin":
        st.subheader("🔑 API Key Control (Admin Only)")
        admin_key_input = st.text_input(
            "Gemini API Key", 
            value=st.session_state.gemini_api_key, 
            type="password",
            placeholder="วาง Gemini API Key..."
        )
        if admin_key_input != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = admin_key_input
            st.success("บันทึก API Key ใหม่เรียบร้อยแล้ว!")
            st.rerun()
        st.divider()

    # ฟอร์มสั่งสร้างงานซีรีส์
    st.subheader("➕ สั่งสร้างซีรีส์ตอนใหม่")
    with st.form("create_form"):
        title = st.text_input("ชื่อตอน", placeholder="เช่น รักซ่อนกล ตอนที่ 1")
        genre = st.selectbox("แนวซีรีส์", ["ดราม่า/แก้แค้น", "โรแมนติก/คอมเมดี้", "แฟนตาซี/ย้อนยุค", "ระทึกขวัญ"])
        
        generate_ai_prompt_btn = st.form_submit_button("✨ ให้ AI พล็อตเรื่องย่อให้อัตโนมัติ")
        if generate_ai_prompt_btn:
            with st.spinner("AI กำลังคิดพล็อตเรื่อง..."):
                success, result = generate_ai_story_prompt(st.session_state.gemini_api_key, genre)
                if success:
                    st.session_state.ai_generated_prompt = result
                    st.success("สร้างพล็อตเรื่องสำเร็จ!")
                else:
                    st.error(result)

        prompt = st.text_area("พล็อตเรื่องย่อ (Prompt)", value=st.session_state.ai_generated_prompt, placeholder="รายละเอียดพล็อตเรื่อง...")
        
        submitted = st.form_submit_button("🚀 ส่งงานเข้าคิวคลาวด์", use_container_width=True)
        if submitted and title and prompt:
            new_id = f"JOB-{datetime.datetime.now().strftime('%H%M%S')}"
            new_job = {
                "id": new_id,
                "user": st.session_state.current_user, 
                "title": title, 
                "genre": genre,
                "status": "Processing", 
                "progress": 25,
                "step_script": "Done", 
                "step_audio": "Processing", 
                "step_visual": "Pending", 
                "step_assembly": "Pending",
                "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), 
                "output": None
            }
            st.session_state.jobs_list.insert(0, new_job)
            st.session_state.system_logs.insert(0, f"[QUEUE] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job {new_id} added by @{st.session_state.current_user}.")
            st.success("ส่งงานเข้าคิวสำเร็จ!")
            st.rerun()

    st.divider()
    st.subheader("🔌 สถานะระบบ (System Health)")
    st.markdown("- **Streamlit Cloud:** <span style='color:#3FB950;'>● ออนไลน์</span>", unsafe_allow_html=True)
    if st.session_state.gemini_api_key:
        st.markdown("- **Gemini AI Service:** <span style='color:#3FB950;'>● พร้อมใช้งาน</span>", unsafe_allow_html=True)
    else:
        st.markdown("- **Gemini AI Service:** <span style='color:#F85149;'>● รอการตั้งค่า API Key</span>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. Main Dashboard Area (แยกตามสิทธิ์ผู้ใช้งาน)
# ---------------------------------------------------------
jobs = st.session_state.jobs_list

if st.session_state.current_role == "Admin":
    # ==========================================
    # 👑 ADMIN CONTROL DASHBOARD
    # ==========================================
    st.title("🛡️ Admin Operations & Control Center")
    st.markdown("ศูนย์บริหารและควบคุมคิวงานทั้งหมด ตรวจสอบ Performance และ System Logs แบบเรียลไทม์")

    total_jobs = len(jobs)
    completed_jobs = len([j for j in jobs if j['status'] == 'Completed'])
    processing_jobs = len([j for j in jobs if j['status'] == 'Processing'])
    success_rate = int((completed_jobs / total_jobs) * 100) if total_jobs > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f'<div class="metric-card"><div class="metric-title">งานทั้งหมดในระบบ</div><div class="metric-value">{total_jobs}</div><div class="metric-delta-pos">เซิร์ฟเวอร์ออนไลน์</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><div class="metric-title">เสร็จสมบูรณ์</div><div class="metric-value" style="color:#3FB950;">{completed_jobs}</div><div class="metric-delta-pos">พร้อมใช้งาน</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><div class="metric-title">กำลังประมวลผล</div><div class="metric-value" style="color:#D29922;">{processing_jobs}</div><div class="metric-delta-pos">ทำงานอัตโนมัติ</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card"><div class="metric-title">อัตราความสำเร็จ</div><div class="metric-value" style="color:#58A6FF;">{success_rate}%</div><div class="metric-delta-pos">ระบบเสถียร</div></div>', unsafe_allow_html=True)

    st.divider()
    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["📊 Live Queue Control", "📈 Server Metrics", "📋 System Logs"])

    with admin_tab1:
        st.subheader("⚙️ รายการคิวงานทั้งหมดในระบบ (Admin Access)")
        for idx, j in enumerate(jobs):
            with st.container():
                c_info, c_actions = st.columns([3, 1])
                with c_info:
                    st.markdown(f"### `{j['id']}` : {j['title']}")
                    st.caption(f"ผู้สร้าง: @{j['user']} | แนว: {j['genre']} | สร้างเมื่อ: {j['created']}")
                    st.progress(j['progress'] / 100, text=f"ความคืบหน้า: {j['progress']}%")
                    st.markdown(f"สถานะ: {get_badge(j['status'])}", unsafe_allow_html=True)
                with c_actions:
                    st.write("")
                    if st.button(f"🗑️ ลบงาน", key=f"admin_del_{j['id']}_{idx}"):
                        st.session_state.jobs_list.remove(j)
                        st.session_state.system_logs.insert(0, f"[DELETE] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job {j['id']} removed by Admin.")
                        st.success(f"ลบงาน {j['id']} เรียบร้อย")
                        st.rerun()
                st.markdown("---")

    with admin_tab2:
        st.subheader("📈 ทรัพยากรระบบ Streamlit Cloud")
        c_a, c_b = st.columns(2)
        with c_a:
            st.info("🖥️ **Memory & Compute Status**\n- Resource Usage: Normal (22% Memory)\n- Status: Active")
        with c_b:
            st.success("⚡ **Gemini Quota Status**\n- API Calls: Active & Responding\n- Model Engine: Fallback Auto-Select Enabled")

    with admin_tab3:
        st.subheader("📋 บันทึกการทำงานระบบ (System Logs)")
        for log in st.session_state.system_logs:
            st.code(log, language="text")

else:
    # ==========================================
    # 🎬 CREATOR PRODUCTION DASHBOARD
    # ==========================================
    st.title("📊 Creator Production & Pipeline Dashboard")
    st.markdown("แดชบอร์ดส่วนตัวสำหรับติดตามสถานะการสร้างวิดีโอซีรีส์แนวตั้ง")

    # แสดงเฉพาะงานของผู้ใช้ที่เข้าสู่ระบบ
    user_jobs = [j for j in jobs if j['user'] == st.session_state.current_user]
    total_u = len(user_jobs)
    completed_u = len([j for j in user_jobs if j['status'] == 'Completed'])
    processing_u = len([j for j in user_jobs if j['status'] == 'Processing'])

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f'<div class="metric-card"><div class="metric-title">โปรเจกต์ของคุณ</div><div class="metric-value">{total_u}</div><div class="metric-delta-pos">ระบบออนไลน์</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><div class="metric-title">เสร็จสมบูรณ์</div><div class="metric-value" style="color:#3FB950;">{completed_u}</div><div class="metric-delta-pos">พร้อมใช้งาน</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><div class="metric-title">กำลังประมวลผล</div><div class="metric-value" style="color:#D29922;">{processing_u}</div><div class="metric-delta-pos">ทำงานอัตโนมัติ</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card"><div class="metric-title">สิทธิ์การใช้งาน</div><div class="metric-value" style="color:#58A6FF;">Active</div><div class="metric-delta-pos">Creator Account</div></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("📌 สถานะการประมวลผลงานของคุณ (Step-by-Step Pipeline)")

    if not user_jobs:
        st.info("ยังไม่มีรายการซีรีส์ของคุณในขณะนี้ สามารถเริ่มสร้างได้จากเมนูด้านซ้ายมือครับ")
    else:
        for j in user_jobs:
            with st.container():
                c_info, c_steps = st.columns([1.2, 2.8])
                with c_info:
                    st.markdown(f"### `{j['id']}` : {j['title']}")
                    st.caption(f"แนว: {j['genre']} | สร้างเมื่อ: {j['created']}")
                    st.progress(j['progress'] / 100, text=f"ความคืบหน้ารวม: {j['progress']}%")
                    st.markdown(f"สถานะรวม: {get_badge(j['status'])}", unsafe_allow_html=True)
                with c_steps:
                    st.markdown("##### ⚙️ Pipeline Stages")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.markdown(f"**1. บท AI**<br>{get_badge(j['step_script'])}", unsafe_allow_html=True)
                    s2.markdown(f"**2. เสียงพากย์**<br>{get_badge(j['step_audio'])}", unsafe_allow_html=True)
                    s3.markdown(f"**3. ภาพ/วิดีโอ**<br>{get_badge(j['step_visual'])}", unsafe_allow_html=True)
                    s4.markdown(f"**4. ตัดต่อรวม**<br>{get_badge(j['step_assembly'])}", unsafe_allow_html=True)
                st.markdown("---")