
🚧 PotholeAI — Road Health Intelligence System

AI-powered road damage detection & geospatial mapping platform built with Computer Vision and Deep Learning.

🌍 Overview

PotholeAI is a smart road infrastructure monitoring system that uses a Convolutional Neural Network (CNN) to detect potholes from road images and map them in real time.

The platform combines:

🧠 Computer Vision (CNN-based classification)

🗺️ Geospatial mapping (Folium + GPS integration)

📊 Severity scoring & confidence analytics

📱 Mobile-first responsive UI

📥 Automated CSV reporting

This project is designed to scale from a web-based prototype to a vehicle-integrated road safety system and smart city infrastructure solution.

🚀 Features
🔍 AI-Based Detection

CNN model trained to classify:

Pothole

Normal Road

Confidence scoring

Severity classification (Low / Medium / High)

🗺️ Live Interactive Map

Auto pin placement using geocoded location

Severity-based marker colors

Persistent session map

Real-time popup details

📊 Analytics Dashboard

Total images scanned

Pothole count

Normal road count

Average confidence score

📂 Report Generation

Download scan results as CSV

Download full session history

Includes:

Filename

Prediction

Severity

Location

GPS coordinates

Timestamp

📱 Mobile-Optimized UI

Camera upload support

Fully responsive layout

Dark theme infrastructure dashboard design

🏗️ Tech Stack
Layer	Technology
ML Model	TensorFlow, Keras
Web Framework	Streamlit
Mapping	Folium
Geolocation	Geopy (Nominatim)
Data Handling	Pandas
UI Styling	Custom CSS
🧠 Model Architecture

The CNN model uses:

Conv2D Layers

MaxPooling Layers

Dense Layers

Sigmoid activation (binary classification)

Binary Crossentropy loss

Adam optimizer

Input size: 128x128x3

📸 How It Works

User enters road location OR coordinates.

Location is geocoded to latitude & longitude.

User uploads road images.

CNN processes each image.

Model predicts pothole probability.

Severity is calculated from confidence.

Pins appear on live map.

Results can be downloaded as CSV.

🛠️ Installation

Clone the repository:

git clone https://github.com/yourusername/PotholeAI.git
cd PotholeAI

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
📁 Project Structure
PotholeAI/
│
├── app.py
├── cnn_model.py
├── pothole_cnn_model.h5
├── dataset/
│   ├── pothole/
│   └── normal/
├── requirements.txt
└── README.md
📈 Future Roadmap

This project is designed to evolve into a scalable infrastructure intelligence platform.

Planned upgrades:

🔴 Real-time video detection

🚗 Vehicle-integrated hardware module

☁️ Cloud database for permanent map storage

📊 Municipal analytics dashboard

🤖 Multi-class detection (Cracks, Waterlogging, Surface Damage)

🛰️ Drone-based road scanning

📡 GPS auto-capture mobile app

🔐 User authentication & role-based access

Long-term vision:

AI-powered Smart Road Infrastructure Monitoring System for Smart Cities.

💡 Potential Use Cases

Municipal road maintenance departments

Smart city projects

Infrastructure inspection agencies

Fleet management companies

Road safety research

🎯 Vision

PotholeAI aims to bridge the gap between AI and public infrastructure by creating a scalable, automated road damage intelligence platform.

The long-term goal is to enable real-time road health monitoring systems that improve public safety and infrastructure efficiency.

👩‍💻 Author

Ayushi Shukla
Front-End & Machine Learning Developer
Building AI for social impact and smart infrastructure.

📜 License

This project is open-source for educational and research purposes.

🚀 What This README Does

This makes your project look:

Professional

Scalable

Startup-oriented

Technically strong

Vision-driven

Not just “CNN pothole classifier”.
