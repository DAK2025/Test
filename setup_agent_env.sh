#!/bin/bash

# AI Agent Environment Setup Script
# This script sets up everything you need to start building AI agents

echo "🤖 Setting up AI Agent Development Environment..."
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Found Python $PYTHON_VERSION"

# Create project directory
PROJECT_NAME=${1:-"my-ai-agent"}
echo "📁 Creating project directory: $PROJECT_NAME"
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv agent-env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source agent-env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing dependencies..."
pip install openai-agents openai python-dotenv fastapi uvicorn aiohttp

# Create project structure
echo "🏗️ Creating project structure..."
mkdir -p {src,examples,tools,agents,tests,logs}

# Create .env file
echo "🔐 Creating environment file..."
cat > .env << 'EOF'
# OpenAI API Configuration
OPENAI_API_KEY=your-api-key-here

# Agent Configuration
DEFAULT_MODEL=gpt-4o-mini
MAX_TOKENS=1000
TEMPERATURE=0.7

# Logging
LOG_LEVEL=INFO
EOF

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Environment
.env
agent-env/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/*.log
*.log

# OS
.DS_Store
Thumbs.db

# Agent specific
agent_logs.jsonl
conversation_history.json
EOF

# Create basic agent example
echo "🤖 Creating basic agent example..."
cat > src/basic_agent.py << 'EOF'
#!/usr/bin/env python3
"""
Basic AI Agent Example
Run this to test your setup!
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from agents import Agent, Runner
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False
    import openai

async def test_openai_sdk():
    """Test with OpenAI Agent SDK"""
    print("🧪 Testing OpenAI Agent SDK...")
    
    agent = Agent(
        name="Test Agent",
        instructions="You are a helpful test assistant. Respond briefly and clearly."
    )
    
    result = await Runner.run(agent, "Say hello and confirm you're working!")
    print(f"✅ Agent Response: {result.final_output}")

async def test_basic_openai():
    """Test basic OpenAI API"""
    print("🧪 Testing basic OpenAI API...")
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and confirm the API is working!"}
        ],
        max_tokens=100
    )
    
    print(f"✅ API Response: {response.choices[0].message.content}")

async def main():
    print("🚀 Testing AI Agent Setup")
    print("=" * 40)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your-api-key-here':
        print("❌ Please set your OpenAI API key in the .env file")
        print("   Edit .env and replace 'your-api-key-here' with your actual API key")
        return
    
    try:
        if FRAMEWORK_AVAILABLE:
            await test_openai_sdk()
        else:
            await test_basic_openai()
        
        print("\n🎉 Setup successful! Your AI agent environment is ready.")
        print("\nNext steps:")
        print("1. Check out the examples/ directory")
        print("2. Run: python examples/basic_openai_agent.py")
        print("3. Run: python examples/from_scratch_agent.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your OpenAI API key is valid")
        print("3. Ensure you have sufficient API credits")

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Create requirements.txt
echo "📋 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core AI Agent Dependencies
openai-agents>=0.2.0
openai>=1.0.0
python-dotenv>=1.0.0

# Web framework (optional)
fastapi>=0.104.0
uvicorn>=0.24.0

# Additional utilities
aiohttp>=3.9.0
requests>=2.31.0
pydantic>=2.0.0

# Development tools (optional)
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
EOF

# Create README for the project
echo "📖 Creating project README..."
cat > README.md << 'EOF'
# My AI Agent Project

This project contains AI agent implementations using various approaches.

## Quick Start

1. **Set your OpenAI API Key**:
   ```bash
   # Edit .env file and add your API key
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

2. **Activate the environment**:
   ```bash
   source agent-env/bin/activate
   ```

3. **Test the setup**:
   ```bash
   python src/basic_agent.py
   ```

4. **Run examples**:
   ```bash
   python examples/basic_openai_agent.py
   python examples/from_scratch_agent.py
   ```

## Project Structure

```
my-ai-agent/
├── agent-env/          # Virtual environment
├── src/                # Main source code
├── examples/           # Example implementations
├── tools/              # Custom tools
├── agents/             # Agent definitions
├── tests/              # Unit tests
├── logs/               # Log files
├── .env                # Environment variables
└── requirements.txt    # Dependencies
```

## Examples

- `examples/basic_openai_agent.py` - OpenAI Agent SDK examples
- `examples/from_scratch_agent.py` - Custom agent implementation
- `src/basic_agent.py` - Simple test agent

## Development

1. Add new tools in the `tools/` directory
2. Create specialized agents in the `agents/` directory
3. Add tests in the `tests/` directory
4. Check logs in the `logs/` directory

## Resources

- [OpenAI Agent SDK Documentation](https://openai.github.io/openai-agents-python/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Project Guide](../README.md)
EOF

# Create a simple tool example
echo "🔧 Creating example tool..."
mkdir -p tools
cat > tools/example_tools.py << 'EOF'
"""
Example tools for AI agents
Add your custom tools here!
"""

import requests
from typing import Dict, Any
from datetime import datetime

def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> Dict[str, Any]:
    """Calculate tip and total for a bill.
    
    Args:
        bill_amount: The bill amount in dollars
        tip_percentage: Tip percentage (default: 15%)
    
    Returns:
        Dictionary with tip amount and total
    """
    tip_amount = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip_amount
    
    return {
        "bill_amount": f"${bill_amount:.2f}",
        "tip_percentage": f"{tip_percentage}%",
        "tip_amount": f"${tip_amount:.2f}",
        "total": f"${total:.2f}"
    }

def word_count(text: str) -> Dict[str, int]:
    """Count words, characters, and lines in text.
    
    Args:
        text: The text to analyze
    
    Returns:
        Dictionary with counts
    """
    words = len(text.split())
    characters = len(text)
    lines = len(text.split('\n'))
    
    return {
        "words": words,
        "characters": characters,
        "lines": lines
    }

# Add more tools here...
EOF

echo "✅ Environment setup complete!"
echo ""
echo "📍 Location: $(pwd)"
echo "🐍 Python: $PYTHON_VERSION"
echo "📦 Virtual environment: agent-env/"
echo ""
echo "🚀 Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Run: source agent-env/bin/activate"
echo "3. Test: python src/basic_agent.py"
echo "4. Explore: python examples/basic_openai_agent.py"
echo ""
echo "📚 Documentation: See README.md for more details"
echo "🎉 Happy agent building!"