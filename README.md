# How to Build Your Own AI Agent System

This comprehensive guide will walk you through building AI agent systems for your environment. We'll cover multiple approaches from using existing frameworks to building from scratch.

## Table of Contents

1. [Understanding AI Agents](#understanding-ai-agents)
2. [Quick Start with OpenAI Agent SDK](#quick-start-with-openai-agent-sdk)
3. [Building from Scratch (No Frameworks)](#building-from-scratch)
4. [Framework Comparison](#framework-comparison)
5. [Production Considerations](#production-considerations)
6. [Advanced Patterns](#advanced-patterns)

## Understanding AI Agents

AI agents are systems that can:
- **Perceive** their environment (receive inputs)
- **Reason** about the information (process and understand)
- **Act** using tools and functions (execute actions)
- **Learn** and adapt over time (improve performance)

### Key Components

1. **LLM Core**: The language model that processes and understands
2. **Tools/Functions**: External capabilities the agent can use
3. **Memory**: Conversation history and context management
4. **Orchestration**: Logic for coordinating multiple agents
5. **Safety**: Guardrails and validation mechanisms

## Quick Start with OpenAI Agent SDK

The easiest way to get started is with OpenAI's official Agent SDK.

### Installation

```bash
# Create virtual environment
python -m venv agent-env
source agent-env/bin/activate  # On Windows: agent-env\Scripts\activate

# Install the SDK
pip install openai-agents

# Set your API key
export OPENAI_API_KEY=your-api-key-here
```

### Simple Agent

```python
from agents import Agent, Runner

# Create a basic agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant that provides clear, concise answers."
)

# Run the agent
async def main():
    result = await Runner.run(agent, "What is artificial intelligence?")
    print(result.final_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Agent with Tools

```python
from agents import Agent, Runner, function_tool
from typing import Dict, Any
import json

@function_tool
async def calculate(expression: str) -> str:
    """Calculate mathematical expressions safely.
    
    Args:
        expression: The mathematical expression to evaluate
    """
    try:
        # Simple safe evaluation (extend for production use)
        result = eval(expression.replace('^', '**'))
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@function_tool
async def get_weather(city: str) -> Dict[str, Any]:
    """Get weather information for a city.
    
    Args:
        city: The name of the city
    """
    # Mock weather data - replace with real API
    return {
        "city": city,
        "temperature": "22°C",
        "condition": "Sunny",
        "humidity": "65%"
    }

# Create agent with tools
agent = Agent(
    name="Tool Assistant",
    instructions="You can help with calculations and weather information. Use the available tools when needed.",
    tools=[calculate, get_weather]
)

async def main():
    result = await Runner.run(agent, "What's 15 * 23 + 7?")
    print(result.final_output)
    
    result = await Runner.run(agent, "What's the weather in Paris?")
    print(result.final_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Multi-Agent System with Handoffs

```python
from agents import Agent, Runner

# Specialized agents
math_agent = Agent(
    name="Math Specialist",
    handoff_description="Expert in mathematical calculations and problem solving",
    instructions="You are a mathematics expert. Solve problems step by step and explain your reasoning.",
    tools=[calculate]
)

weather_agent = Agent(
    name="Weather Specialist", 
    handoff_description="Expert in weather information and forecasting",
    instructions="You provide weather information and forecasts. Be detailed and helpful.",
    tools=[get_weather]
)

# Coordinator agent
coordinator = Agent(
    name="Coordinator",
    instructions="You coordinate between specialists. Route math questions to Math Specialist and weather questions to Weather Specialist.",
    handoffs=[math_agent, weather_agent]
)

async def main():
    result = await Runner.run(coordinator, "Calculate 50 * 80 + 15, then tell me the weather in Tokyo")
    print(result.final_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Building from Scratch

For maximum control and understanding, you can build an agent system from scratch.

### Basic Agent Architecture

```python
import openai
import json
import re
from typing import Dict, Any, Callable, List
from dataclasses import dataclass

@dataclass
class ToolResult:
    success: bool
    result: Any
    error: str = None

class SimpleAgent:
    def __init__(self, name: str, instructions: str, tools: Dict[str, Callable] = None):
        self.name = name
        self.instructions = instructions
        self.tools = tools or {}
        self.conversation_history = []
        
        # Initialize OpenAI client
        self.client = openai.OpenAI()
    
    def add_tool(self, name: str, func: Callable, description: str):
        """Add a tool to the agent"""
        self.tools[name] = {
            'function': func,
            'description': description
        }
    
    def _extract_tool_call(self, response: str) -> tuple:
        """Extract tool calls from LLM response"""
        # Simple pattern matching for tool calls
        tool_pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        match = re.search(tool_pattern, response)
        
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            try:
                # Parse arguments (simplified - extend for production)
                args = json.loads(f'[{args_str}]')
                return tool_name, args
            except:
                return None, None
        
        return None, None
    
    def _execute_tool(self, tool_name: str, args: List) -> ToolResult:
        """Execute a tool with given arguments"""
        if tool_name not in self.tools:
            return ToolResult(False, None, f"Tool '{tool_name}' not found")
        
        try:
            result = self.tools[tool_name]['function'](*args)
            return ToolResult(True, result)
        except Exception as e:
            return ToolResult(False, None, str(e))
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions"""
        prompt = f"{self.instructions}\n\n"
        
        if self.tools:
            prompt += "Available tools:\n"
            for name, tool in self.tools.items():
                prompt += f"- {name}: {tool['description']}\n"
            prompt += "\nTo use a tool, respond with: TOOL_CALL: tool_name(arg1, arg2, ...)\n"
        
        return prompt
    
    async def process_message(self, message: str) -> str:
        """Process a user message and return response"""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        max_iterations = 5  # Prevent infinite loops
        
        for _ in range(max_iterations):
            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                *self.conversation_history
            ]
            
            # Get response from LLM
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message.content
            
            # Check if LLM wants to use a tool
            tool_name, args = self._extract_tool_call(assistant_message)
            
            if tool_name:
                # Execute tool
                tool_result = self._execute_tool(tool_name, args)
                
                # Add tool execution to conversation
                if tool_result.success:
                    tool_message = f"Tool {tool_name} executed successfully. Result: {tool_result.result}"
                else:
                    tool_message = f"Tool {tool_name} failed: {tool_result.error}"
                
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": assistant_message
                })
                self.conversation_history.append({
                    "role": "user", 
                    "content": tool_message
                })
                
                # Continue the loop to get final response
                continue
            else:
                # No tool call, this is the final response
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": assistant_message
                })
                return assistant_message
        
        return "Maximum iterations reached. Please try again."

# Example usage
def calculate(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        result = eval(expression.replace('^', '**'))
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def get_time() -> str:
    """Get current time"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def main():
    # Create agent
    agent = SimpleAgent(
        name="Calculator Agent",
        instructions="You are a helpful assistant that can perform calculations and tell time."
    )
    
    # Add tools
    agent.add_tool("calculate", calculate, "Perform mathematical calculations")
    agent.add_tool("get_time", get_time, "Get the current date and time")
    
    # Interactive loop
    print("Agent ready! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        response = await agent.process_message(user_input)
        print(f"Agent: {response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Advanced Multi-Agent System

```python
import asyncio
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

class MessageType(Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    HANDOFF = "handoff"

@dataclass
class Message:
    type: MessageType
    sender: str
    content: str
    metadata: Dict[str, Any] = None

class AgentOrchestrator:
    def __init__(self):
        self.agents: Dict[str, SimpleAgent] = {}
        self.current_agent: Optional[str] = None
        self.conversation_log: List[Message] = []
    
    def register_agent(self, agent: SimpleAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.name] = agent
    
    def set_active_agent(self, agent_name: str):
        """Set the currently active agent"""
        if agent_name in self.agents:
            self.current_agent = agent_name
        else:
            raise ValueError(f"Agent '{agent_name}' not found")
    
    async def route_message(self, message: str, sender: str = "user") -> str:
        """Route a message to the appropriate agent"""
        # Add message to log
        self.conversation_log.append(Message(
            type=MessageType.USER,
            sender=sender,
            content=message
        ))
        
        # If no current agent, use a simple routing logic
        if not self.current_agent:
            self.current_agent = self._route_to_agent(message)
        
        # Process with current agent
        agent = self.agents[self.current_agent]
        response = await agent.process_message(message)
        
        # Log agent response
        self.conversation_log.append(Message(
            type=MessageType.AGENT,
            sender=self.current_agent,
            content=response
        ))
        
        return response
    
    def _route_to_agent(self, message: str) -> str:
        """Simple routing logic based on keywords"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['calculate', 'math', 'equation']):
            return 'math_agent'
        elif any(word in message_lower for word in ['weather', 'temperature', 'forecast']):
            return 'weather_agent'
        else:
            return 'general_agent'

# Example specialized agents
async def create_specialized_agents():
    # Math agent
    math_agent = SimpleAgent(
        name="math_agent",
        instructions="You are a mathematics expert. Solve problems step by step."
    )
    math_agent.add_tool("calculate", calculate, "Perform mathematical calculations")
    
    # Weather agent
    weather_agent = SimpleAgent(
        name="weather_agent", 
        instructions="You provide weather information."
    )
    
    def mock_weather(city: str) -> str:
        return f"The weather in {city} is sunny, 25°C"
    
    weather_agent.add_tool("get_weather", mock_weather, "Get weather for a city")
    
    # General agent
    general_agent = SimpleAgent(
        name="general_agent",
        instructions="You are a helpful general assistant."
    )
    
    return [math_agent, weather_agent, general_agent]

async def main():
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Create and register agents
    agents = await create_specialized_agents()
    for agent in agents:
        orchestrator.register_agent(agent)
    
    # Interactive session
    print("Multi-agent system ready! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        response = await orchestrator.route_message(user_input)
        print(f"Agent: {response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Framework Comparison

| Framework | Complexity | Learning Curve | Production Ready | Best For |
|-----------|------------|----------------|------------------|----------|
| OpenAI Agent SDK | Low-Medium | Easy | High | Quick development, OpenAI integration |
| Custom/Scratch | High | Steep | Depends | Maximum control, specific requirements |
| AutoGen | Medium-High | Medium | Medium | Microsoft ecosystem, complex workflows |
| LangChain | High | Steep | Medium | Complex chains, many integrations |
| Swarm (Legacy) | Medium | Medium | Low | Learning, experimentation |

## Production Considerations

### 1. Error Handling & Resilience

```python
import logging
from functools import wraps
import time

def retry_with_backoff(max_retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
            
        return wrapper
    return decorator

@retry_with_backoff()
async def robust_api_call(agent, message):
    return await agent.process_message(message)
```

### 2. Security & Validation

```python
import re
from typing import Any

class SecurityValidator:
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Remove potentially harmful content"""
        # Remove potential injection attempts
        dangerous_patterns = [
            r'<script.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*='
        ]
        
        for pattern in dangerous_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def validate_tool_args(args: Any) -> bool:
        """Validate tool arguments"""
        if isinstance(args, str):
            return len(args) < 1000  # Reasonable length limit
        elif isinstance(args, (list, tuple)):
            return len(args) < 10 and all(len(str(arg)) < 100 for arg in args)
        return True

# Usage in agent
class SecureAgent(SimpleAgent):
    async def process_message(self, message: str) -> str:
        # Sanitize input
        clean_message = SecurityValidator.sanitize_input(message)
        
        # Process normally
        return await super().process_message(clean_message)
```

### 3. Monitoring & Logging

```python
import json
from datetime import datetime
from typing import Dict, Any

class AgentMonitor:
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'tool_usage': {},
            'response_times': []
        }
    
    def log_request(self, agent_name: str, message: str, response: str, 
                   response_time: float, success: bool):
        """Log agent interaction"""
        
        self.metrics['total_requests'] += 1
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
            
        self.metrics['response_times'].append(response_time)
        
        # Log to file
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'message': message,
            'response': response,
            'response_time': response_time,
            'success': success
        }
        
        with open('agent_logs.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_response_time = sum(self.metrics['response_times']) / len(self.metrics['response_times']) if self.metrics['response_times'] else 0
        
        return {
            **self.metrics,
            'success_rate': self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1),
            'average_response_time': avg_response_time
        }
```

### 4. Scalable Deployment

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

class MessageRequest(BaseModel):
    message: str
    agent_name: str = None

class MessageResponse(BaseModel):
    response: str
    agent_name: str
    success: bool

app = FastAPI(title="AI Agent API")

# Global orchestrator
orchestrator = AgentOrchestrator()
monitor = AgentMonitor()

@app.on_startup
async def startup():
    """Initialize agents on startup"""
    agents = await create_specialized_agents()
    for agent in agents:
        orchestrator.register_agent(agent)

@app.post("/chat", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest):
    """Main chat endpoint"""
    try:
        start_time = time.time()
        
        if request.agent_name:
            orchestrator.set_active_agent(request.agent_name)
        
        response = await orchestrator.route_message(request.message)
        
        response_time = time.time() - start_time
        monitor.log_request(
            orchestrator.current_agent,
            request.message,
            response,
            response_time,
            True
        )
        
        return MessageResponse(
            response=response,
            agent_name=orchestrator.current_agent,
            success=True
        )
        
    except Exception as e:
        monitor.log_request("error", request.message, str(e), 0, False)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get agent performance metrics"""
    return monitor.get_metrics()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Advanced Patterns

### 1. Memory and Context Management

```python
from typing import List, Dict
import json

class ConversationMemory:
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict]] = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim history if too long
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for session"""
        return self.conversations.get(session_id, [])
    
    def clear_session(self, session_id: str):
        """Clear conversation history for session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
```

### 2. Tool Registry and Dynamic Loading

```python
import importlib
import inspect
from typing import Callable, Dict, Any

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, name: str, func: Callable, description: str, 
                    category: str = "general"):
        """Register a tool"""
        signature = inspect.signature(func)
        
        self.tools[name] = {
            'function': func,
            'description': description,
            'category': category,
            'signature': str(signature),
            'parameters': list(signature.parameters.keys())
        }
    
    def load_tools_from_module(self, module_path: str):
        """Dynamically load tools from a module"""
        module = importlib.import_module(module_path)
        
        for name in dir(module):
            obj = getattr(module, name)
            if hasattr(obj, '_is_agent_tool'):
                self.register_tool(
                    name,
                    obj,
                    getattr(obj, '_description', ''),
                    getattr(obj, '_category', 'general')
                )
    
    def get_tools_by_category(self, category: str) -> Dict[str, Any]:
        """Get tools by category"""
        return {name: tool for name, tool in self.tools.items() 
                if tool['category'] == category}

# Decorator for marking tools
def agent_tool(description: str, category: str = "general"):
    def decorator(func):
        func._is_agent_tool = True
        func._description = description
        func._category = category
        return func
    return decorator

# Example tool module (tools/math_tools.py)
@agent_tool("Add two numbers", "math")
def add(a: float, b: float) -> float:
    return a + b

@agent_tool("Calculate factorial", "math")
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

### 3. Agent Workflows and Pipelines

```python
from typing import List, Callable, Any
from dataclasses import dataclass

@dataclass
class WorkflowStep:
    name: str
    agent: SimpleAgent
    condition: Callable[[Any], bool] = None
    input_transform: Callable[[Any], str] = None
    output_transform: Callable[[str], Any] = None

class AgentWorkflow:
    def __init__(self, name: str):
        self.name = name
        self.steps: List[WorkflowStep] = []
        self.context: Dict[str, Any] = {}
    
    def add_step(self, step: WorkflowStep):
        """Add a step to the workflow"""
        self.steps.append(step)
    
    async def execute(self, initial_input: Any) -> Any:
        """Execute the workflow"""
        current_input = initial_input
        
        for step in self.steps:
            # Check condition if present
            if step.condition and not step.condition(current_input):
                continue
            
            # Transform input if needed
            if step.input_transform:
                agent_input = step.input_transform(current_input)
            else:
                agent_input = str(current_input)
            
            # Execute agent
            response = await step.agent.process_message(agent_input)
            
            # Transform output if needed
            if step.output_transform:
                current_input = step.output_transform(response)
            else:
                current_input = response
            
            # Update context
            self.context[f"{step.name}_output"] = current_input
        
        return current_input

# Example usage
async def create_research_workflow():
    # Create agents
    researcher = SimpleAgent("researcher", "You research topics and gather information")
    analyzer = SimpleAgent("analyzer", "You analyze information and draw conclusions")
    writer = SimpleAgent("writer", "You write clear, concise reports")
    
    # Create workflow
    workflow = AgentWorkflow("research_pipeline")
    
    workflow.add_step(WorkflowStep(
        name="research",
        agent=researcher,
        input_transform=lambda x: f"Research this topic: {x}"
    ))
    
    workflow.add_step(WorkflowStep(
        name="analyze", 
        agent=analyzer,
        input_transform=lambda x: f"Analyze this research: {x}"
    ))
    
    workflow.add_step(WorkflowStep(
        name="write",
        agent=writer,
        input_transform=lambda x: f"Write a report based on: {x}"
    ))
    
    return workflow
```

## Getting Started Recommendations

1. **For Beginners**: Start with OpenAI Agent SDK for quick results
2. **For Learning**: Build a simple agent from scratch to understand the concepts
3. **For Production**: Use OpenAI Agent SDK with proper monitoring and security
4. **For Complex Systems**: Consider custom solutions with proper architecture

## Environment Setup Script

Save this as `setup_agent_env.sh`:

```bash
#!/bin/bash

# Create project directory
mkdir -p my-ai-agent
cd my-ai-agent

# Create virtual environment
python -m venv agent-env
source agent-env/bin/activate

# Install dependencies
pip install openai-agents openai python-dotenv fastapi uvicorn

# Create basic structure
mkdir -p {src,tools,agents,tests}

# Create environment file
cat > .env << EOF
OPENAI_API_KEY=your-api-key-here
EOF

# Create basic agent
cat > src/basic_agent.py << 'EOF'
from agents import Agent, Runner
import asyncio

agent = Agent(
    name="MyAgent",
    instructions="You are a helpful assistant."
)

async def main():
    result = await Runner.run(agent, "Hello, how are you?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
EOF

echo "Environment setup complete!"
echo "1. Add your OpenAI API key to .env file"
echo "2. Run: python src/basic_agent.py"
```

This guide provides multiple pathways to building AI agents, from simple implementations to production-ready systems. Choose the approach that best fits your needs and technical requirements.
