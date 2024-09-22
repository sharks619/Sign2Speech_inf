# 손말이음 - 청각장애인을 위한 AI 기반 실시간 의사소통 서비스 (Inference)

![Sonmalyieum Overview](./image/손말이음.png)  
*Figure: Sonmalyieum Project Overview and Key Features*

## Overview

**손말이음 (Sonmalyieum)** is an AI-based real-time communication service that provides seamless communication between hearing-impaired and non-impaired individuals. This project supports three core functionalities: **Sign2Speech**, **Speech2Sign**, and **Multi-party Communication**.

This repository focuses on the **Sign2Speech** feature, providing the necessary scripts and tools for running inference using the pre-trained model. This feature translates Korean Sign Language (KSL) into both text and speech in real-time.

## Model Information

- The **Sign2Speech** feature is powered by a deep learning model trained on the **KSL-GUIDE** dataset, which is a large-scale dataset specifically designed for Korean Sign Language (KSL).
- The pre-trained model provided here is capable of recognizing key KSL gestures and translating them into text and speech.

## Input Options
You can provide either:
- **Live webcam feed**: By modifying the `input_source` in the code to use your webcam.
- **Pre-recorded video**: By specifying the path to the input video file.

## Output
The output will include:
- Translated text in real-time based on KSL gestures.
- Optionally, the translated text will be converted into synthesized speech using a text-to-speech engine. In this experiment, we utilized the **ChatGPT API** to generate text-to-speech output, providing a smooth and natural voice for real-time communication.
