# ใช้ Python 3.10 รุ่นพรีเมียมขนาดเล็ก
FROM python:3.10-slim

# กำหนดโฟลเดอร์ทำงานในคอนเทนเนอร์
WORKDIR /app

# คัดลอกไฟล์ dependencies และติดตั้ง
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดทั้งหมดลงในคอนเทนเนอร์
COPY . .

# เปิดพอร์ต 8080 สำหรับ Cloud Run
EXPOSE 8080

# คำสั่งรัน FastAPI ด้วย Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]