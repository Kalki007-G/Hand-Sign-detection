# 🤟 ASL Hand Sign Detection with Voice Output 🔊

A real-time American Sign Language (ASL) hand gesture detection system using Computer Vision and Python that not only displays the recognized letters but also **speaks them aloud** using **Google Text-to-Speech (gTTS)**.

---

## 📌 Project Objective

This project aims to recognize ASL hand signs using a webcam and convert the detected signs into corresponding English alphabets, and then **speak them out loud** for accessibility using voice synthesis.

---

## 📷 Demo

![demo-gif](demo/demo.gif)  
*(Include your own demo gif or video if available)*

---

## 🛠️ Tech Stack

| Tool/Library     | Purpose                              |
|------------------|--------------------------------------|
| Python 3.x       | Core programming language            |
| OpenCV           | Video capture and display            |
| MediaPipe        | Real-time hand landmark detection    |
| NumPy            | Array processing                     |
| gTTS             | Converts detected letter to speech   |
| Playsound        | Plays the generated speech file      |

---

## 🔍 Features

- 📹 Real-time webcam-based hand sign detection
- 🔤 Displays the corresponding ASL alphabet
- 🗣️ Uses `gTTS` to speak detected letters/words
- 🎯 Lightweight and easy to use

---

## 🧠 How It Works

1. Captures video frames using webcam.
2. Detects hand landmarks using **MediaPipe**.
3. Classifies the hand sign into a corresponding **ASL letter**.
4. Converts the detected letter to voice using **Google Text-to-Speech (gTTS)**.
5. Plays the spoken letter using **Playsound** or similar audio library.

---

## 🚀 Getting Started

### 📦 Installation

```bash
pip install -r requirements.txt
