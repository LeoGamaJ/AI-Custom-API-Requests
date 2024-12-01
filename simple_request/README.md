# 🤖 LLM Chat Collection

A comprehensive collection of Python interfaces for interacting with state-of-the-art language model APIs.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🌟 Features

### 🔥 Multiple LLM Support
- **Google Gemini**: Access to Gemini 1.5 models (Pro, Flash, Flash-8B)
- **Groq**: Interface with LLaMA, Mixtral, and Gemma models
- **OpenAI**: Integration with GPT-4 and GPT-3.5
- **Anthropic**: Support for Claude models
- **Perplexity**: Access to pplx-7b-online and pplx-70b-online models

### 📝 Common Features
- Interactive terminal-based chat
- Conversation history management
- Flexible parameter configuration
- Automatic conversation saving
- Robust error handling
- Multi-language support

### 🖼️ Special Features
- **Gemini**: Image analysis and multimodal processing
- **Groq**: Real-time response streaming
- **OpenAI**: Advanced functions and code analysis
- **Anthropic**: Long document processing
- **Perplexity**: Real-time online research

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### API Configuration
Create a `.env` file in the project root with your API keys:
```env
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
PERPLEXITY_API_KEY=your_perplexity_key
```

### Available Scripts

#### 🟦 Gemini (`gemini_req.py`)
```python
from gemini_req import GeminiChat
chat = GeminiChat()
chat.send_message("Hello, how are you?")
```

#### 🟨 Groq (`groq_requests.py`)
```python
from groq_requests import GroqChat
chat = GroqChat()
chat.send_message("Tell me about AI")
```

#### 🟩 OpenAI (`openai_request.py`)
```python
from openai_request import OpenAIChat
chat = OpenAIChat()
chat.send_message("Explain programming")
```

#### 🟪 Anthropic (`anthropic_request.py`)
```python
from anthropic_request import AnthropicChat
chat = AnthropicChat()
chat.send_message("Analyze this text")
```

#### 🟧 Perplexity (`perplexity_request.py`)
```python
from perplexity_request import PerplexityChat
chat = PerplexityChat()
chat.send_message("Research this topic")
```

## 💡 Common Commands

All scripts share similar basic commands:
```bash
# Exit chat
exit, q

# Clear history
clear, cls

# Save conversation
save, s

# Configure model
config model=model-name

# Configure temperature
config temperature=0.7
```

## 🔧 Configuration

| Parameter | Description | Range |
|-----------|-------------|-------|
| model | Model to be used | Varies by API |
| temperature | Response creativity | 0.0 to 1.0 |
| top_p | Nucleus sampling | 0.0 to 1.0 |
| max_tokens | Token limit | Varies by model |

## 📋 Requirements

### Core Dependencies
- requests
- python-dotenv
- Pillow (for Gemini)
- groq (for Groq)
- openai (for OpenAI)
- anthropic (for Anthropic)
- perplexity-python (for Perplexity)

## 🎯 Use Cases

- **Chatbots**: Create custom chat interfaces
- **Text Processing**: Content analysis and generation
- **Image Analysis**: Visual processing (Gemini)
- **Research**: Real-time search (Perplexity)
- **Programming**: Code assistance (OpenAI)
- **Documentation**: Long-form text generation and analysis (Anthropic)

## 🤝 Contributing

1. Fork the project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

Leo Gama
- GitHub: [@LeoGamaJ](https://github.com/LeoGamaJ)
- Email: leo@leogama.cloud 
- LinkedIn: (https://www.linkedin.com/in/leonardo-gama-jardim/)

## 📄 License

This project is licensed under the MIT License.

