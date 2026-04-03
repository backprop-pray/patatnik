# 🌱 Tracky — Autonomous AI Farming Rover

<!-- [Tracky IRL robot final image] -->
![tracky final image](https://github.com/user-attachments/assets/cfc96767-bfc6-4308-ade7-2c8ed000f266)

> 🥇 1st Place @ Hack TUES & TUES Fest  
> Built by **Backprop & Pray**

<!-- [Image with my team holding the 1st place prize and gold medals on the stage] -->
![W W W](https://github.com/user-attachments/assets/a07dd089-7921-4e42-9e31-6e74f22eb0e4)

---

## 🚜 What is Tracky?

**Tracky** is a fully autonomous AI-powered rover that navigates farmland, detects plant diseases, and maps problem areas — helping farmers **save resources, reduce waste, and increase yield**.

It combines:
- 🤖 Reinforcement Learning navigation
- 👁️ Computer Vision (anomaly detection)
- 🧠 Knowledge Graph + RAG system
- 📱 Mobile app with real-time insights

---

## 🌍 The Problem

- 🌱 **~1 billion tons** of crops are lost globally due to preventable plant diseases  
- 🚜 Existing solutions are:
  - Expensive  
  - Not fully autonomous  
  - Hard to scale for small farms  

- 👨‍🌾 **75–85% of farms worldwide are small-scale**
  - Limited access to advanced technology
  - High cost of machinery
  - Inefficient use of pesticides

---

## 💡 Our Solution

Tracky enables:

- 🔍 **Early disease detection**
- 🎯 **Targeted treatment (decimeter precision)**
- 💰 **Lower operational costs**
- 🌿 **Reduced chemical waste**

It’s designed to be:
- **Affordable** (prototype < €50 excluding compute)
- **Compact & terrain-adaptive**
- **Fully autonomous**

---

## 🧠 System Overview

<!-- [Image with 3d printed parts and embedded diagram bundled as 1 image] -->
<img width="1141" height="634" alt="embedded image" src="https://github.com/user-attachments/assets/8e6c5c01-536f-4031-8dd0-d6237eef4e0e" />

### 🔩 Embedded System

- Raspberry Pi-based control unit  
- Distance sensors for navigation  
- GPS module for precise mapping  
- High-resolution camera for plant monitoring  
- Custom 3D-printed chassis (assembled in 1 day)

---

### 🤖 AI Navigation (Reinforcement Learning)

- Custom RL agent controls the rover  
- Learns terrain dynamically  
- Fully autonomous movement between crop rows  

<!-- [Cool image of the robot around plants] -->
![cool imave v2 as png](https://github.com/user-attachments/assets/1441c786-f4bf-49b8-a42a-6ac1a975daaf)

---

### 👁️ Vision System (Anomaly Detection)

<!-- [Gif for plant disease vision detection] -->
![good video](https://github.com/user-attachments/assets/1acc1c00-834d-48d7-85fb-4d06c4dac6bb)


- Learns **farm-specific “healthy baseline”**
- Detects deviations → potential diseases
- Continuously improves with new data

Output:
- 📍 GPS-tagged anomaly
- 📊 Confidence + classification

---

### 🧠 Knowledge System (RAG + Graph)

<!-- [3 pictograms about knowledge graph bundled as 1 image] -->
<img width="988" height="442" alt="graph pictograms" src="https://github.com/user-attachments/assets/4bddaff0-a004-40a3-8f32-4f5e59d2cc1b" />

- Connects:
  - Plant conditions  
  - Historical cases  
  - Treatments  

- Enables:
  - Smart recommendations  
  - Continuous learning across farms  

---

### 📱 Mobile App

<!-- [3 iOS app screenshots bundled as a single image] -->
<img width="1139" height="634" alt="mobile app image" src="https://github.com/user-attachments/assets/55929576-0062-49bf-ac1c-f5ad9d518d83" />

- Live map of detected issues  
- Field analytics  
- Feedback loop → improves models  

---

## 🔄 Self-Improving System

All components work together:

- RL agent improves navigation  
- Vision model improves detection  
- Knowledge system improves recommendations  

➡️ With every farm, every user, every day — **Tracky gets better**

---

## 💼 Business Model

Tracky is not just a robot.

It’s a **data + intelligence platform**:

- 💸 Low-cost rover (easy adoption)
- 📊 Subscription for:
  - Monitoring
  - Analytics
  - Disease prediction
- 🌍 Data platform for researchers & bio labs

---

## 👨‍🌾 Built by Backprop & Pray

<!-- [Image of the team dressed as farmers with our gold medals] -->
<img width="1536" height="1024" alt="us the farmers" src="https://github.com/user-attachments/assets/d0a19e3d-a5de-4828-b6f7-d7d62ea23ef1" />

We are a team of students from TUES, building real-world AI systems from scratch.
