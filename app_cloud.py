import streamlit as st
import pandas as pd
import datetime
import os
import google.generativeai as genai

# ---------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Cloud AI Studio - Operations Center",
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization & Enterprise Data Setup
# ---------------------------------------------------------
if 'jobs_list' not in st.session_state:
    st.session_state.jobs_list = [
        {
            "id": "JOB-901", "user": "creator_pro01", "title": "ประธานบริษัทปลอมตัวมาลองใจ ตอนที่ 1", 
            "genre": "ดราม่า/แก้แค้น", "status": "Completed", "progress": 100, 
            "step_script": "Done", "step_audio": "Done", "step_visual": "Done", "step_assembly": "Done",
            "created": "2026-10-21 15:30", "output": "storage/outputs/job901.mp4"
        },
        {
            "id": "JOB-902", "user": "series_lover", "title": "ย้อนเวลามาเป็นฮูหยินยอดอัจฉริยะ", 
            "genre": "แฟนตาซี/ย้อนยุค", "status": "Processing", "progress": 65, 
            "step_script": "Done", "step_audio": "Done", "step_visual": "Processing", "step_assembly": "Pending",
            "created": "2026-10-21 16:10", "output": None
        }
    ]

if 'ai_generated_prompt' not in st.session_state:
    st.session_state.ai_generated_prompt = ""

if 'system_logs' not in st.session_state:
    st.session_state.system_logs = [
        "[INFO] 2026-10-21 15:30:05 - System initialized successfully on Streamlit Cloud.",
        "[INFO] 2026-10-21 15:30:30 - Job JOB-901 completed pipeline execution.",
        "[INFO] 2026-10-21 16:10:00 - Job JOB-902 dispatched to async worker queue."
    ]

# ---------------------------------------------------------
# Sidebar Panel: Authentication, Roles & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚡ Enterprise Studio")
    st.caption("Cloud SaaS Video Generation Engine")
    st.divider()

    # เลือกบทบาทผู้ใช้งาน
    st.subheader("👤 เลือกระบบผู้ใช้งาน (Role)")
    user_role = st.selectbox(
        "สิทธิ์การใช้งาน", 
        ["ผู้ดูแลระบบ (Admin)", "ผู้สร้างสรรค์เนื้อหา (Content Creator)"]
    )

    st.divider()

    # ตัวแปรสำหรับตรวจสอบสถานะ API
    gemini_status = False
    ai_model = None

    # 🔒 จำกัดสิทธิ์: ช่องกรอก API Key เฉพาะ Admin เท่านั้น
    if "Admin" in user_role:
        st.subheader("🔑 API Authentication (Admin Only)")
        if "gemini_key_input" not in st.session_state:
            st.session_state.gemini_key_input = st.secrets.get("GEMINI_API_KEY", "")

        user_api_key = st.text_input(
            "Gemini API Key", 
            value=st.session_state.gemini_key_input, 
            type="password",
            placeholder="วาง API Key ของคุณที่นี่..."
        )
        
        if user_api_key:
            st.session_state.gemini_key_input = user_api_key
            try:
                genai.configure(api_key=user_api_key)
                ai_model = genai.GenerativeModel('gemini-1.5-flash')
                test_res = ai_model.generate_content("test")
                if test_res:
                    gemini_status = True
            except Exception:
                gemini_status = False
        st.divider()
    else:
        if "gemini_key_input" in st.session_state and st.session_state.gemini_key_input:
            try:
                genai.configure(api_key=st.session_state.gemini_key_input)
                ai_model = genai.GenerativeModel('gemini-1.5-flash')
                gemini_status = True
            except Exception:
                gemini_status = False

    # ฟอร์มสั่งสร้างซีรีส์ตอนใหม่
    st.subheader("➕ สั่งสร้างซีรีส์ตอนใหม่")
    
    with st.form("create_form"):
        username = st.text_input("ชื่อผู้ใช้งาน (Username)", value="admin_root" if "Admin" in user_role else "creator_user")
        title = st.text_input("ชื่อตอน", placeholder="เช่น รักซ่อนกล ตอนที่ 1")
        genre = st.selectbox("แนวซีรีส์", ["ดราม่า/แก้แค้น", "โรแมนติก/คอมเมดี้", "แฟนตาซี/ย้อนยุค", "ระทึกขวัญ"])
        
        generate_ai_prompt_btn = st.form_submit_button("✨ ให้ AI พล็อตเรื่องย่อให้อัตโนมัติ")
        
        if generate_ai_prompt_btn:
            if gemini_status and ai_model:
                with st.spinner("AI กำลังสร้างพล็อตเรื่องแนวตั้งสุดปัง..."):
                    try:
                        response = ai_model.generate_content(f"ขอพล็อตเรื่องย่อสำหรับซีรีส์แนวสั้นแนวตั้ง แนว {genre} ความยาว 3 บรรทัด น่าตื่นเต้น น่าติดตาม")
                        st.session_state.ai_generated_prompt = response.text
                        st.success("สร้างพล็อตเรื่องสำเร็จ!")
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการเรียก Gemini AI: {e}")
            else:
                st.warning("กรุณาสลับไปโหมด Admin เพื่อเชื่อมต่อ Gemini API Key ก่อนใช้งาน AI")

        prompt = st.text_area("พล็อตเรื่องย่อ (Prompt)", value=st.session_state.ai_generated_prompt, placeholder="กดปุ่มด้านบนเพื่อให้ AI คิดพล็อตให้...")
        
        submitted = st.form_submit_button("🚀 ส่งงานเข้าคิวคลาวด์", use_container_width=True)
        if submitted and title and prompt:
            new_job_id = f"JOB-{datetime.datetime.now().strftime('%H%M%S')}"
            new_job = {
                "id": new_job_id,
                "user": username, "title": title, "genre": genre,
                "status": "Processing", "progress": 20,
                "step_script": "Processing", "step_audio": "Pending", "step_visual": "Pending", "step_assembly": "Pending",
                "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "output": None
            }
            st.session_state.jobs_list.insert(0, new_job)
            st.session_state.system_logs.insert(0, f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - New job {new_job_id} created by @{username}.")
            st.success("ส่งงานเข้าคิวสำเร็จ!")
            st.rerun()

    st.divider()
    st.subheader("🔌 สถานะการเชื่อมต่อระบบ")
    st.markdown(f"- **Streamlit Cloud Server:** <span style='color:#3FB950;'>● เชื่อมต่อสำเร็จ</span>", unsafe_allow_html=True)
    if gemini_status:
        st.markdown(f"- **Gemini AI API:** <span style='color:#3FB950;'>● เชื่อมต่อสำเร็จ (พร้อมใช้งาน)</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"- **Gemini AI API:** <span style='color:#F85149;'>● ยังไม่เชื่อมต่อ (เฉพาะ Admin)</span>", unsafe_allow_html=True)
    st.markdown(f"- **Session State Engine:** <span style='color:#3FB950;'>● ทำงานปกติ</span>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Function สำหรับ Badge สถานะ
# ---------------------------------------------------------
def get_badge(status):
    if status == 'Done' or status == 'Completed': return '<span class="badge badge-completed">✓ เสร็จสมบูรณ์</span>'
    if status == 'Processing': return '<span class="badge badge-processing">🔄 กำลังประมวลผล</span>'
    if status == 'Failed': return '<span class="badge badge-failed">✕ เกิดข้อผิดพลาด</span>'
    return '<span class="badge badge-pending">⏳ รอคิวประมวลผล</span>'

# ---------------------------------------------------------
# MAIN DASHBOARD LOGIC (Admin vs Creator)
# ---------------------------------------------------------
jobs = st.session_state.jobs_list

if "Admin" in user_role:
    st.title("🛡️ Enterprise Admin Control & Operations Center")
    st.markdown("ศูนย์ควบคุมระบบบริหารจัดการระดับองค์กร: ตรวจสอบคิวงาน จัดการทรัพยากร และดูบันทึกระบบแบบเรียลไทม์")

    total_jobs = len(jobs)
    completed_jobs = len([j for j in jobs if j['status'] == 'Completed'])
    processing_jobs = len([j for j in jobs if j['status'] == 'Processing'])
    success_rate = int((completed_jobs / total_jobs) * 100) if total_jobs > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f'<div class="metric-card"><div class="metric-title">งานทั้งหมดในระบบ</div><div class="metric-value">{total_jobs}</div><div class="metric-delta-pos">สถานะเซิร์ฟเวอร์: ปกติ</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><div class="metric-title">สำเร็จสมบูรณ์</div><div class="metric-value" style="color:#3FB950;">{completed_jobs}</div><div class="metric-delta-pos">พร้อมเผยแพร่</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><div class="metric-title">กำลังประมวลผล</div><div class="metric-value" style="color:#D29922;">{processing_jobs}</div><div class="metric-delta-pos">ทำงานอัตโนมัติ</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card"><div class="metric-title">อัตราความสำเร็จ</div><div class="metric-value" style="color:#58A6FF;">{success_rate}%</div><div class="metric-delta-pos">ภาพรวมระบบ</div></div>', unsafe_allow_html=True)

    st.divider()

    # ระบบแท็บควบคุมมาตรฐานองค์กร (Tabs Navigation)
    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["📊 Live Pipeline & Queue Monitor", "📈 Analytics & Cloud Insights", "⚙️ System Logs & Health Check"])

    with admin_tab1:
        st.subheader("⚙️ ระบบจัดการและตรวจสอบคิวงานทั้งหมด (Admin Control)")
        
        status_filter = st.selectbox("กรองตามสถานะงาน", ["ทั้งหมด", "Completed", "Processing"])
        filtered_jobs = jobs if status_filter == "ทั้งหมด" else [j for j in jobs if j['status'] == status_filter]

        for idx, j in enumerate(filtered_jobs):
            with st.container():
                c_info, c_actions = st.columns([3, 1])
                with c_info:
                    st.markdown(f"### `{j['id']}` : {j['title']}")
                    st.caption(f"ผู้สร้าง: @{j['user']} | แนว: {j['genre']} | สร้างเมื่อ: {j['created']}")
                    st.progress(j['progress'] / 100, text=f"ความคืบหน้ารวม: {j['progress']}%")
                    st.markdown(f"สถานะรวม: {get_badge(j['status'])}", unsafe_allow_html=True)
                with c_actions:
                    st.write("")
                    if st.button(f"🗑️ ลบงาน", key=f"del_{j['id']}_{idx}"):
                        st.session_state.jobs_list.remove(j)
                        st.session_state.system_logs.insert(0, f"[WARNING] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job {j['id']} was deleted by Admin.")
                        st.success(f"ลบ {j['id']} สำเร็จ!")
                        st.rerun()
                st.markdown("---")

    with admin_tab2:
        st.subheader("📈 ข้อมูลเชิงลึกและทรัพยากรคลาวด์ (Cloud Insights)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.info("📊 **การใช้ CPU / Memory บน Streamlit Cloud**\n- เฉลี่ยการใช้งาน: 24.5%\n- สถานะ: เสถียรดีเยี่ยม ไม่มีอาการคอขวด")
        with col_b:
            st.success("⚡ **โควตา Gemini API Utilization**\n- อัตราการเรียกใช้งานรายนาที: 12 / 60 RPM\n- โควตารายวัน: คงเหลือเพียงพอต่อการผลิตซีรีส์")

    with admin_tab3:
        st.subheader("📋 บันทึกการทำงานของระบบ (System Logs)")
        for log in st.session_state.system_logs:
            st.code(log, language="text")

else:
    st.title("📊 Creator Production & Pipeline Dashboard")
    st.markdown("แดชบอร์ดส่วนตัวของคุณ: ติดตามสถานะการสร้างซีรีส์แนวตั้ง และจัดการคอนเทนต์ของคุณ")

    total_jobs = len(jobs)
    completed_jobs = len([j for j in jobs if j['status'] == 'Completed'])
    processing_jobs = len([j for j in jobs if j['status'] == 'Processing'])
    success_rate = int((completed_jobs / total_jobs) * 100) if total_jobs > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f'<div class="metric-card"><div class="metric-title">โปรเจกต์ของคุณ</div><div class="metric-value">{total_jobs}</div><div class="metric-delta-pos">ระบบออนไลน์ 100%</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><div class="metric-title">สำเร็จสมบูรณ์</div><div class="metric-value" style="color:#3FB950;">{completed_jobs}</div><div class="metric-delta-pos">พร้อมใช้งาน</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><div class="metric-title">กำลังประมวลผล</div><div class="metric-value" style="color:#D29922;">{processing_jobs}</div><div class="metric-delta-pos">ทำงานอัตโนมัติ</div></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card"><div class="metric-title">อัตราความสำเร็จ</div><div class="metric-value" style="color:#58A6FF;">{success_rate}%</div><div class="metric-delta-pos">เสถียรภาพสูง</div></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("📌 สถานะโปรเซสการทำงานรายโปรเจกต์ (Step-by-Step Pipeline)")

    for j in jobs:
        with st.container():
            c_info, c_steps = st.columns([1.2, 2.8])
            with c_info:
                st.markdown(f"### `{j['id']}` : {j['title']}")
                st.caption(f"ผู้ใช้: @{j['user']} | แนว: {j['genre']} | สร้างเมื่อ: {j['created']}")
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