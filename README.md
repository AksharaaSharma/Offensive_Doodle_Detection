# 🕵️‍♀️ Doodle Detector

<img src="/api/placeholder/800/400" alt="Doodle Detector Banner" />

## About
Doodle Detector is an intelligent drawing analysis tool that uses advanced AI to detect potentially offensive content in hand-drawn sketches. By combining the power of CLIP (Contrastive Language-Image Pre-Training) and Google's Gemini models, this application provides real-time feedback on doodle content with detailed explanations.

## ✨ Features

- **Interactive Drawing Canvas**: Easily create doodles with multiple drawing tools
- **Upload Capability**: Analyze existing drawings from your device
- **Dual AI Analysis**: 
  - Fine-tuned CLIP model for fast initial classification
  - Google Gemini 1.5 for detailed content breakdown
- **Comprehensive Results**: Get confidence scores and specific element identification
- **Beautiful Interface**: Engaging design with responsive elements

## 🚀 Technology Stack

- **Streamlit**: Interactive web application framework
- **PyTorch & Transformers**: Powers the CLIP model implementation
- **Google Generative AI**: Integration with Gemini 1.5 for in-depth content analysis
- **PIL & NumPy**: Image processing capabilities
- **Streamlit Drawable Canvas**: Interactive drawing functionality

## 📸 Screenshots

<div style="display: flex; justify-content: space-between;">
<img src="/api/placeholder/400/300" alt="Drawing Interface" />
<img src="/api/placeholder/400/300" alt="Analysis Results" />
</div>

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/doodle-detector.git
cd doodle-detector

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Streamlit
- Google Generative AI
- PIL
- NumPy

## 💡 How It Works

1. **Draw or Upload**: Create your doodle directly in the app or upload an existing image
2. **AI Processing**: The system runs your drawing through both CLIP and Gemini models
3. **Analysis**: Get a combined assessment with classification and detailed explanation
4. **Review**: Understand exactly why content may be flagged as potentially offensive

## 🔒 Privacy & Ethics

Doodle Detector is designed with privacy and ethical considerations at its core:
- All processing happens in real-time
- No user drawings are stored permanently
- Transparent explanations for all classifications
- Educational focus to help understand content guidelines

## 👩‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- OpenAI for the CLIP model architecture
- Google for the Gemini API
- Streamlit team for the amazing web framework
- All contributors who helped shape this project

---

<p align="center">Made with ❤️ for responsible AI development</p>
