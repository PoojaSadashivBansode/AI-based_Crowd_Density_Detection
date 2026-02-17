# 🚀 Complete Guide: Running Dashboard in Colab with ngrok

## Why ngrok is Better

| Feature | ngrok ✅ | localtunnel ❌ |
|---------|---------|----------------|
| **Stability** | Very stable | Often disconnects |
| **Speed** | Fast | Slow |
| **Setup** | One-time token | IP verification every time |
| **URL Quality** | Clean URLs | Random URLs |
| **Production-ready** | Yes | No |

---

## 🎯 Quick Start

### **Step 1: Get ngrok Token** (One-time, 2 minutes)

1. Go to: https://dashboard.ngrok.com/signup
2. Sign up (free, just email)
3. Copy your auth token from: https://dashboard.ngrok.com/get-started/your-authtoken
4. Save it somewhere (you'll need it once)

---

### **Step 2: Open Colab Notebook**

1. Go to: https://colab.research.google.com
2. **File** → **Open notebook** → **GitHub** tab
3. Enter: `PoojaSadashivBansode/AI-based_Crowd_Density_Detection`
4. Select: **colab_notebook.ipynb** ⭐ NEW

---

### **Step 3: Enable GPU**

1. **Runtime** → **Change runtime type**
2. Select **T4 GPU**
3. Click **Save**

---

### **Step 4: Run Setup Cells**

Run cells in order:

**Cell 1:** Check GPU
```python
!nvidia-smi
```

**Cell 2:** Clone & Install
```python
# Automatically installs everything including pyngrok
```

**Cell 3:** Upload video.mp4
- Use Files sidebar (📁 icon)
- Upload your `video.mp4`

---

### **Step 5: Choose Your Option**

## **OPTION 1: Batch Processing** ⚡ (Recommended)

**Run Cell 5:**
```python
!python main.py --source video.mp4 --threshold 50 --yolo yolov8s.pt
```

**Best for:**
- Processing full videos
- Maximum GPU speed
- Generating reports

**Output:**
- `output.mp4` (annotated video)
- `crowd_data.csv` (detailed logs)

**View results:** Run cells 6 & 7

---

## **OPTION 2: Live Dashboard** 🎛️ (For Demos)

**Run Cell 9:**
```python
# Dashboard with ngrok
```

**Before running:**
1. Find this line in the cell:
   ```python
   NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"
   ```
2. Replace with your actual token:
   ```python
   NGROK_TOKEN = "2abCdEfGhIjKlMnOpQrStUvWxYz_1234567890"
   ```

**Then run the cell!**

**Output:**
```
✅ Dashboard is live!
🔗 Access your dashboard here:
   https://abc123.ngrok-free.app
```

**Click the URL** → Dashboard opens → Click **"🚀 Start Monitoring"**

**Features:**
- Real-time crowd counting
- Multi-factor switching visualization
- Live charts & alerts
- Processes 100 frames (Colab optimization)

---

## 📊 Comparison

| Feature | Batch Processing | ngrok Dashboard |
|---------|------------------|-----------------|
| **Speed** | Fastest | Interactive |
| **GPU Usage** | 100% | ~70% |
| **Output** | Video + CSV | Live UI |
| **Setup** | Just upload video | Need ngrok token |
| **Best For** | Analysis | Demos |

---

## 🛠️ Troubleshooting

### **"ngrok token invalid"**
- Copy token again from https://dashboard.ngrok.com/get-started/your-authtoken
- Make sure no extra spaces

### **"video.mp4 not found"**
- Upload to Files sidebar (📁) on left
- Make sure filename is exactly `video.mp4`

### **"CSRNet counts too high"**
- Latest code has calibration fix (0.18)
- Run `!git pull` in a new cell before processing

### **Dashboard stuck on "Starting"**
- Click "🚀 Start Monitoring" button
- Refresh the page
- Check video.mp4 is uploaded

### **Slow dashboard updates**
- This is normal with tunneling
- Use batch processing for speed

---

## 💡 Pro Tips

1. **Save your ngrok token** - You only need to get it once
2. **Use batch for analysis** - Much faster for processing
3. **Use dashboard for demos** - Great for showing real-time
4. **Download CSV** - Right-click `crowd_data.csv` → Download
5. **Reuse tokens** - Same ngrok token works forever

---

## 📝 Quick Commands Reference

### In Colab:
```python
# Batch processing (fastest)
!python main.py --source video.mp4 --threshold 50 --yolo yolov8s.pt

# Pull latest code
!git pull

# Check GPU
!nvidia-smi

# View logs
!cat crowd_data.csv | head -20
```

---

## 🎯 Recommended Workflow

For best results:

1. **First time:** Set up ngrok, get token
2. **Analysis:** Use batch processing
3. **Demo:** Use ngrok dashboard
4. **Local development:** Run `streamlit run app.py` on PC

---

## ✅ What You Get

### With Batch Processing:
- **output.mp4** - Annotated video with counts, bounding boxes, model info
- **crowd_data.csv** - Frame-by-frame logs with switching reasons

### With Dashboard:
- **Live monitoring** - Real-time crowd counting
- **Interactive UI** - Adjust threshold, see charts
- **Visual feedback** - Model switching, alerts

---

**You're all set!** 🎉

Files in repo:
- `colab_notebook.ipynb` ⭐ Use this one for ngrok
- `app_colab.py` - Optimized dashboard code
- `main.py` - Batch processing script
- `app.py` - Full dashboard (for local use)
