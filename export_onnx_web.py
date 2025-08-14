# export_onnx_web.py
# ใช้ครั้งเดียวเพื่อสร้าง ONNX สำหรับเว็บ + classes.json
# pip install ultralytics onnx onnxsim

from ultralytics import YOLO
import os, json, shutil, sys

# ----- ปรับ path โมเดล .pt ของคุณ -----
PT_PATH = 'Detecting-Cheating-with-computer-vision-Real-Time-Exam-Hall-Monitoring/best.pt'
IMG_SIZE = 640

# ----- ตำแหน่ง output บนเว็บ -----
WEB_MODELS_DIR = os.path.join('web', 'models')  # ปรับตามโครงสร้างโปรเจกต์คุณ
os.makedirs(WEB_MODELS_DIR, exist_ok=True)

print(f'Loading model: {PT_PATH}')
model = YOLO(PT_PATH)

# สร้าง ONNX แบบเหมาะกับ WebGPU: nms=False, half=False (FP32), simplify=True
print('Exporting ONNX for web (nms=False, fp32)...')
onnx_path = model.export(
    format='onnx',
    opset=17,
    imgsz=IMG_SIZE,
    simplify=True,
    half=False,      # ใช้ FP32 เพื่อความเข้ากันได้สูงสุดบน onnxruntime-web
    dynamic=False,
    nms=False        # <— เอา NMS ออก ให้เว็บทำ NMS เอง (เร็ว/เสถียรกว่า)
)

# onnx_path อาจเป็น path หรือ list; handle ให้ครอบคลุม
if isinstance(onnx_path, (list, tuple)):
    onnx_path = next((p for p in onnx_path if str(p).endswith('.onnx')), None)

if not onnx_path or not os.path.exists(onnx_path):
    raise SystemExit('❌ Export failed: ONNX file not found')

# ย้าย/คัดลอกไปโฟลเดอร์เว็บ
dst_onnx = os.path.join(WEB_MODELS_DIR, 'best_webgpu.onnx')
shutil.copy2(onnx_path, dst_onnx)
print(f'✅ ONNX saved to: {dst_onnx}')

# สร้าง classes.json จาก names ที่ฝังใน checkpoint
names = getattr(model.model, 'names', None) or getattr(model, 'names', None)
if isinstance(names, dict):
    names = [names[i] for i in range(len(names))]
if not isinstance(names, (list, tuple)):
    print('⚠️  No class names found in checkpoint; classes.json will be skipped')
else:
    classes_path = os.path.join(WEB_MODELS_DIR, 'classes.json')
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump(list(names), f, ensure_ascii=False, indent=2)
    print(f'✅ classes.json saved to: {classes_path}')

print('Done.')
