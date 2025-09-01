# Introduction

This rust crate implements inference of Nvidia's new Parakeet-tdt-0.6B v2 model using onnxruntime.

# How to run

1. Clone the repository

```bash
git clone https://github.com/jason-ni/parakeet-rs.git
```

2. Download onnx model from release page

```bash
wget https://github.com/jason-ni/parakeet-rs/releases/download/v0.1.0/parakeet-tdt-0.6b-v2-onnx.tar.gz
tar xvf parakeet-tdt-0.6b-v2-onnx.tar.gz
```

3. Run the example

```bash
cargo run --example run -- <onnx model dir> true
```

# Advertisement

This project is part of my personal ASR centric application called **听风转录**. In this application, I bundled Whisper.cpp/SenseVoice/Parakeet 
into a single package. Specific in Parakeet, I recently added Parakeet-tdt_ctc-0.6B-ja(Japanese model) and Parakeet-tdt-0.6B-V3(25 European languages model). 
It supports both Windows and MacOS. You can download it from https://pan.quark.cn/s/15d095d09465
