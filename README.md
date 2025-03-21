# TuneIT

A user-friendly web interface built with Streamlit that enables easy fine-tuning of models on custom datasets.

## Overview

TuneIT simplifies the process of adapting  models to specific domains and tasks. This tool democratizes access to advanced model customization, allowing users without extensive machine learning expertise to leverage the power of Gemma for their unique applications.

## Features

**Dataset Management**
- Support for multiple data formats (CSV, JSONL, TXT)
- Built-in validation and preprocessing tools
- Optional data augmentation capabilities
- Sample dataset templates for common fine-tuning tasks

**Intuitive Hyperparameter Configuration**
- Sensible defaults for quick starts
- Advanced options for experienced users
- Interactive tooltips explaining each parameter's purpose
- Presets for common fine-tuning scenarios (classification, generation, etc.)

**Real-time Training Visualization**
- Dynamic loss and accuracy curves
- Live training metrics dashboard
- Generated text samples during training
- Resource utilization monitoring

**Flexible Model Export Options**
- Download fine-tuned models in multiple formats:
  - PyTorch (.pt)
  - TensorFlow SavedModel
  - GGUF for efficient local inference
  - Hugging Face compatible format

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gemma-finetuning-ui.git
cd gemma-finetuning-ui

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run finetune.py
```

## Requirements

- Python 3.8+
- Streamlit 1.10+
- PyTorch 2.0+ or TensorFlow 2.8+
- Transformers 4.25+
- 16GB+ RAM (32GB+ recommended for larger models)
- CUDA-compatible GPU (8GB+ VRAM) for efficient training

## Usage Guide

### 1. Prepare Your Dataset

Format your data as:
- CSV files with text and label columns
- JSONL with prompt/completion pairs
- Text files with appropriate delimiters

### 2. Upload and Configure

- Upload your dataset through the interface
- Select preprocessing options
- Choose the Gemma model variant (2B or 7B)
- Configure memory and compute constraints

### 3. Set Hyperparameters

Adjust key parameters:
- Learning rate
- Batch size
- Training epochs
- Output sequence length
- Optimization algorithm

### 4. Train and Monitor

- Start training with a single click
- Monitor real-time metrics
- Pause/resume functionality for long training runs
- Early stopping options based on validation metrics

### 5. Export Your Model

- Download in your preferred format
- Get integration code snippets for different frameworks
- Access deployment instructions

## Project Structure

```
gemma-finetuning-ui/
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
└── README.md                 # Detailed documentation
```

## Contributing

We welcome contributions to improve the Gemma Fine-tuning UI! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- Google for releasing the Gemma family of open models
- The Streamlit team for their excellent framework
- The open-source ML community for their invaluable contributions

## Contact

For questions or support, please open an issue on GitHub or contact the project maintainers.

---
Answer from Perplexity: pplx.ai/share
