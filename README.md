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
