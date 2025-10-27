# visionguardai
It's a system that works on cctv cams of roads , traffics , tolls etc and read realtime video data and detect for certain activities that may lead to accident, like no helmets, using smartphones, rashdriving, riding on wrong lane, yawning etc using YOLOv8 trained with custom datasets and use OPENCV for interpreting realtime cam data. 
# VisionGuard AI 🔒🧠
**Real-time Helmet Safety Monitoring System with AI + Voice Alerts**

VisionGuard AI is a powerful computer-vision safety solution designed to automatically detect **riders without helmets** using live CCTV or webcam feeds. The system instantly highlights violations and issues **text + voice alerts** to prevent road accidents before they happen.

---

## 🚨 Why VisionGuard AI?

Every year, thousands of road deaths occur due to non-helmet use.  
VisionGuard AI provides:

✅ Real-time violation detection  
✅ Continuous monitoring without human supervision  
✅ Accurate detection with advanced AI  
✅ Alerts for immediate corrective actions  

---

## 🧠 Under the Hood — AI Technology

| Component | Details |
|----------|---------|
| **AI Model** | YOLOv8n (Ultralytics) |
| **Model Type** | Deep Learning – Object Detection |
| **Training Dataset** | COCO + Custom Helmet Detection Dataset |
| **Hardware Acceleration** | Runs on CPU / GPU |
| **Alerts** | On-screen text with beeping |
| **Video Source** | Webcam feed (cv2.VideoCapture) |

### 🎯 What does YOLOv8 detect?

✔ Helmet  
❌ No Helmet  
🚦 Motorbike + Rider context  
🛑 Unsafe behavior flagging  

YOLOv8 is one of the fastest and most accurate real-time vision systems in the world — ideal for surveillance deployments.

---

## 📂 Project Structure

