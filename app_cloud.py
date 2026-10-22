import streamlit as st
import pandas as pd
import datetime

# ---------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Cloud AI Studio - Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background-color: #0B0E14;
        color: #E6EDF3;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .metric-title {
        color: #8B949E;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #F0F6FC;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 8px;
    }
    .metric-delta-pos {
        color: #3FB950;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 5px;
    }

    /* Badges */
    .badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
        display: inline-block;
    }
    .badge-completed { background-color: #113222; color: #3FB950; border: 1px solid #1B472C; }
    .badge-processing { background-color: #38290D; color: #D29922; border: 1px solid #573E0E; }
    .badge-failed { background-color: #3E1B1B; color: #F85149; border: 1px solid #5C2828; }
    .badge-pending { background-color: #21262D; color: #8B949E; border: 1px solid #30363D; }

    /* Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #11151C;
        border-right: 1px solid #30363D;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Mock Data State (จำลองข้อมูลคิวงานและระบบคลาวด์)
# ---------------------------------------------------------
if 'jobs' not in st.session_state:
    st.session_state.jobs = [
        {
            "id": "JOB-901", "user": "creator_pro01", "title": "ประธานบริษัทปลอมตัวมาลองใจ ตอนที่ 1", 
            "genre": "ดราม่า/แก้แค้น", "status": "Completed", "progress": 100, 
            "step_script": "Done", "step_audio": "Done", "step_visual": "Done", "step_assembly": "Done",
            "created": "2026-10-21 15:30", "output": "storage/outputs/job901.mp4"
        },
        {
            "id": "JOB-902", "user": "series_lover", "title": "ย้อนเวลามาเป็นฮูหยินยอดอัจฉริยะ", 
            "genre": "ย้อนยุค/โรแมนติก", "status": "Processing", "progress": 65, 
            "step_script": "Done", "step_audio": "Done", "step_visual": "Processing", "step_assembly": "Pending",
            "created": "2026-10-21 16:10", "output": None
        },
        {
            "id": "JOB-903", "user": "admin_test", "title": "ระบบลับพลิกชะตาชีวิต", 
            "genre": "แฟนตาซี", "status": "Pending", "progress": 10, 
            "step_script": "Processing", "step_audio": "Pending", "step_visual": "Pending", "step_assembly": "Pending",
            "created": "2026-10-21 16:25", "output": None
        }
    ]

# ---------------------------------------------------------
# Sidebar Panel: Actions & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚡ Enterprise Studio")
    st.caption("Cloud SaaS Video Generation Engine")
    st.divider()

    st.subheader("➕ สั่งสร้างซีรีส์ตอนใหม่")
    with st.form("admin_create_form", clear_on_submit=True):
        username = st.text_input("ชื่อผู้ใช้งาน (Username)", value="admin_root")
        title = st.text_input("ชื่อตอน", placeholder="เช่น รักซ่อนกล ตอนที่ 1")
        genre = st.selectbox("แนวซีรีส์", ["ดราม่า/แก้แค้น", "โรแมนติก/คอมเมดี้", "แฟนตาซี/ย้อนยุค", "ระทึกขวัญ"])
        prompt = st.text_area("พล็อตเรื่องย่อ (Prompt)", placeholder="เช่น พระเอกถูกหักหลังจากคนใกล้ตัว...")
        
        submitted = st.form_submit_button("🚀 ส่งงานเข้าคิวคลาวด์", use_container_width=True)
        if submitted and title and prompt:
            new_id = f"JOB-{len(st.session_state.jobs) + 101}"
            st.session_state.jobs.insert(0, {
                "id": new_id, "user": username, "title": title, 
                "genre": genre, "status": "Processing", "progress": 15, 
                "step_script": "Processing", "step_audio": "Pending", "step_visual": "Pending", "step_assembly": "Pending",
                "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "output": None
            })
            st.success(f"ส่งงาน {new_id} สำเร็จ!")
            st.rerun()

    st.divider()
    st.subheader("☁️ Google Cloud Free Tier Quota")
    st.progress(0.35, text="Cloud Run Requests: 35% (ฟรี)")
    st.progress(0.12, text="Firestore Reads: 12% (ฟรี)")
    st.progress(0.50, text="Gemini API Limits: 50% (15 RPM)")

# ---------------------------------------------------------
# Main Dashboard Layout
# ---------------------------------------------------------
st.title("📊 Cloud AI Production & Operations Dashboard")
st.markdown("ระบบมอนิเตอร์และควบคุมการประมวลผลซีรีส์แนวตั้งอัตโนมัติแบบครบวงจรระดับมืออาชีพ")

# Top KPI Summary Cards
col1, col2, col3, col4 = st.columns(4)

total_jobs = len(st.session_state.jobs)
completed_jobs = len([j for j in st.session_state.jobs if j['status'] == 'Completed'])
processing_jobs = len([j for j in st.session_state.jobs if j['status'] == 'Processing'])
success_rate = int((completed_jobs / total_jobs) * 100) if total_jobs > 0 else 0

col1.markdown(f'<div class="metric-card"><div class="metric-title">คิวงานทั้งหมด (Total Jobs)</div><div class="metric-value">{total_jobs}</div><div class="metric-delta-pos">↑ 12% จากชั่วโมงที่แล้ว</div></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-card"><div class="metric-title">สำเร็จสมบูรณ์ (Completed)</div><div class="metric-value" style="color:#3FB950;">{completed_jobs}</div><div class="metric-delta-pos">พร้อมใช้งาน 100%</div></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="metric-card"><div class="metric-title">กำลังประมวลผล (In Pipeline)</div><div class="metric-value" style="color:#D29922;">{processing_jobs}</div><div class="metric-delta-pos">ระบบทำงานอัตโนมัติ</div></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="metric-card"><div class="metric-title">อัตราความสำเร็จ (Success Rate)</div><div class="metric-value" style="color:#58A6FF;">{success_rate}%</div><div class="metric-delta-pos">เสถียรภาพระบบสูง</div></div>', unsafe_allow_html=True)

st.divider()

# Helper Status Badge
def get_badge(status):
    if status == 'Completed': return '<span class="badge badge-completed">✓ เสร็จสมบูรณ์</span>'
    if status == 'Processing': return '<span class="badge badge-processing">🔄 กำลังประมวลผล</span>'
    if status == 'Failed': return '<span class="badge badge-failed">✕ เกิดข้อผิดพลาด</span>'
    return '<span class="badge badge-pending">⏳ รอคิวประมวลผล</span>'

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📌 Live Pipeline & Queue Monitor", "📈 Analytics & Cloud Insights", "📜 System Logs & Health Check"])

# ---------------------------------------------------------
# TAB 1: Live Pipeline Monitor
# ---------------------------------------------------------
with tab1:
    col_filter_1, col_filter_2 = st.columns([2, 6])
    with col_filter_1:
        status_filter = st.selectbox("กรองตามสถานะ", ["ทั้งหมด", "Processing", "Completed", "Pending"])
    
    st.subheader("สถานะโพรเซสการทำงานรายโปรเจกต์ (Step-by-Step Pipeline)")
    
    filtered_jobs = st.session_state.jobs
    if status_filter != "ทั้งหมด":
        filtered_jobs = [j for j in st.session_state.jobs if j['status'] == status_filter]

    for j in filtered_jobs:
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

# ---------------------------------------------------------
# TAB 2: Analytics & Cloud Insights
# ---------------------------------------------------------
with tab2:
    st.subheader("📈 สถิติการใช้งานระบบและการเติบโต (Usage Analytics)")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("##### จำนวนวิดีโอที่สร้างสำเร็จในแต่ละวัน (Daily Outputs)")
        chart_data = pd.DataFrame({
            'วัน': ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์'],
            'จำนวนวิดีโอ': [12, 19, 15, 22, 30, 45, 40]
        })
        st.bar_chart(chart_data.set_index('วัน'))

    with chart_col2:
        st.markdown("##### สัดส่วนแนวซีรีส์ยอดนิยม (Genre Distribution)")
        genre_data = pd.DataFrame({
            'แนวเรื่อง': ['ดราม่า/แก้แค้น', 'โรแมนติก', 'แฟนตาซี', 'ระทึกขวัญ'],
            'สัดส่วน (%)': [45, 25, 20, 10]
        })
        st.dataframe(genre_data, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 3: System Logs & Health Check
# ---------------------------------------------------------
with tab3:
    st.subheader("🔍 สถานะการเชื่อมต่อ API และ System Logs")
    
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Google Cloud Run", "Online", "Latency: 45ms")
    h2.metric("Firestore DB", "Connected", "Read/Write: Normal")
    h3.metric("Gemini 1.5 API", "Active", "Quota: 50% Free")
    h4.metric("Edge-TTS Service", "Running", "No Errors")

    st.divider()
    st.markdown("##### 🖥️ Live Terminal Stream Logs")
    
    terminal_logs = """
[16:25:01] [System] Initializing Cloud Pipeline Engine...
[16:25:02] [Firestore] Connected successfully to 'ai-series-saas' project database.
[16:25:10] [Gemini-API] Task received for JOB-903. Generating script structure... Done.
[16:25:12] [Edge-TTS] Synthesizing Thai voice model for scene 1/3... Success.
[16:25:15] [Cloud-Storage] Uploading generated assets to Google Cloud Bucket... OK.
[16:25:18] [Pipeline] JOB-902 successfully rendered final vertical video (9:16).
    """
    st.code(terminal_logs, language="bash")
    
    if st.button("🔄 รีเฟรชสถานะแดชบอร์ด"):
        st.rerun()