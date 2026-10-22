from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
import os

app = FastAPI(title="AI Vertical Series Cloud Backend", version="1.0")

# 1. ตั้งค่า Gemini API (ดึงค่าจาก Environment Variable หรือใส่ Key ของคุณ)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
# ใช้ Gemini 1.5 Flash ซึ่งมีความเร็วสูงและอยู่ในโควตาฟรี
ai_model = genai.GenerativeModel('gemini-1.5-flash')

# 2. ตั้งค่าเชื่อมต่อ Google Firestore (ฐานข้อมูลบนคลาวด์)
if not firebase_admin._apps:
    # หากรันบน Cloud Run ระบบจะเชื่อมต่อให้อัตโนมัติด้วย Default Credentials
    firebase_admin.initialize_app()
db = firestore.client()

# โครงสร้างข้อมูลที่รับเข้ามาจากหน้าเว็บ Dashboard
class EpisodeRequest(BaseModel):
    title: str
    genre: str
    prompt_text: str
    username: str = "guest_user"

@app.get("/")
def health_check():
    return {"status": "online", "message": "AI Vertical Series Backend is running smoothly!"}

@app.post("/api/create-episode")
async def create_episode(request: EpisodeRequest, background_tasks: BackgroundTasks):
    try:
        # บันทึกสถานะงานเริ่มต้นลงใน Firestore Database
        doc_ref = db.collection("jobs").document()
        job_id = doc_ref.id
        
        initial_data = {
            "job_id": job_id,
            "username": request.username,
            "title": request.title,
            "genre": request.genre,
            "prompt": request.prompt_text,
            "status": "Pending",
            "progress": 10,
            "step_script": "Pending",
            "step_audio": "Pending",
            "step_visual": "Pending",
            "step_assembly": "Pending",
            "log": "งานถูกส่งเข้าคิวบนระบบคลาวด์แล้ว...",
            "output_url": None,
            "created_at": firestore.SERVER_TIMESTAMP
        }
        doc_ref.set(initial_data)

        # สั่งรันกระบวนการ AI Pipeline แบบ Background Task (ไม่ให้ API หลุดการเชื่อมต่อ)
        background_tasks.add_task(run_cloud_pipeline, job_id, request.prompt_text)

        return {"status": "success", "job_id": job_id, "message": "สร้างคิวงานบนคลาวด์สำเร็จ"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_cloud_pipeline(job_id: str, prompt_text: str):
    doc_ref = db.collection("jobs").document(job_id)
    
    try:
        # --- STEP 1: เขียนบทด้วย Gemini 1.5 Flash ---
        doc_ref.update({
            "status": "Processing",
            "step_script": "Processing",
            "progress": 25,
            "log": "กำลังให้ Google Gemini AI วิเคราะห์และเขียนบทซีรีส์..."
        })
        
        system_prompt = f"""คุณคือผู้กำกับซีรีส์แนวตั้ง 9:16 มืออาชีพ จงเขียนบทซีรีส์ 3 ซีน จากพล็อต: '{prompt_text}' 
ตอบกลับเป็น JSON Array แท้เท่านั้น ห้ามมีข้อความอื่นปน โครงสร้าง:
[
  {{"scene": 1, "dialogue": "บทพูดภาษาไทย", "visual_prompt": "cinematic vertical 9:16 scene description in english"}}
]"""
        
        response = ai_model.generate_content(system_prompt)
        script_result = response.text.strip()
        
        doc_ref.update({
            "step_script": "Done",
            "step_audio": "Processing",
            "step_visual": "Processing",
            "progress": 50,
            "log": "เขียนบทสำเร็จ! กำลังสังเคราะห์เสียงพากย์และสร้างภาพประกอบ..."
        })

        # --- STEP 2 & 3: จำลองกระบวนการเสียง ภาพ และประกอบร่างวิดีโอ ---
        # (บน Cloud Run สามารถเรียกใช้ Edge-TTS และ FFmpeg หรือส่งต่อ Worker ได้ที่นี่)
        import time
        time.sleep(3) # จำลองเวลาประมวลผล

        doc_ref.update({
            "step_audio": "Done",
            "step_visual": "Done",
            "step_assembly": "Done",
            "status": "Completed",
            "progress": 100,
            "log": "🎉 ประมวลผลวิดีโอซีรีส์แนวตั้งเสร็จสมบูรณ์ พร้อมดาวน์โหลด!",
            "output_url": "https://www.w3schools.com/html/mov_bbb.mp4" # ลิงก์ตัวอย่างวิดีโอผลลัพธ์
        })

    except Exception as e:
        doc_ref.update({
            "status": "Failed",
            "step_script": "Failed",
            "log": f"❌ เกิดข้อผิดพลาดในระบบคลาวด์: {str(e)}"
        })