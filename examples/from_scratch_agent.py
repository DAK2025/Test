#!/usr/bin/env python3
"""
From-Scratch AI Agent Implementation
Demonstrates building an AI agent system without frameworks for maximum control and understanding.
"""

import asyncio
import json
import re
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Result of tool execution"""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0

@dataclass
class AgentMessage:
    """Message in agent conversation"""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class SimpleAgent:
    """A simple agent implementation from scratch"""
    
    def __init__(self, name: str, instructions: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[AgentMessage] = []
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        
    def add_tool(self, name: str, func: Callable, description: str, 
                 parameters: Optional[Dict[str, Any]] = None):
        """Add a tool to the agent"""
        self.tools[name] = {
            'function': func,
            'description': description,
            'parameters': parameters or {}
        }
        logger.info(f"Added tool '{name}' to agent '{self.name}'")
    
    def _extract_tool_calls(self, response: str) -> List[Tuple[str, List[Any]]]:
        """Extract tool calls from LLM response using pattern matching"""
        tool_calls = []
        
        # Pattern: TOOL_CALL: function_name(arg1, arg2, ...)
        pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            tool_name = match[0]
            args_str = match[1]
            
            try:
                # Simple argument parsing (extend for production)
                if args_str.strip():
                    # Handle string arguments
                    if args_str.strip().startswith('"') and args_str.strip().endswith('"'):
                        args = [args_str.strip()[1:-1]]
                    else:
                        # Try JSON parsing
                        try:
                            args = json.loads(f'[{args_str}]')
                        except:
                            # Fallback: split by comma
                            args = [arg.strip().strip('"\'') for arg in args_str.split(',')]
                else:
                    args = []
                
                tool_calls.append((tool_name, args))
            except Exception as e:
                logger.warning(f"Failed to parse tool call: {tool_name}({args_str}) - {e}")
                
        return tool_calls
    
    def _execute_tool(self, tool_name: str, args: List[Any]) -> ToolResult:
        """Execute a tool with given arguments"""
        start_time = time.time()
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False, 
                result=None, 
                error=f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}",
                execution_time=time.time() - start_time
            )
        
        try:
            result = self.tools[tool_name]['function'](*args)
            execution_time = time.time() - start_time
            
            logger.info(f"Executed tool '{tool_name}' in {execution_time:.3f}s")
            
            return ToolResult(
                success=True, 
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool '{tool_name}' failed: {str(e)}"
            logger.error(error_msg)
            
            return ToolResult(
                success=False, 
                result=None, 
                error=error_msg,
                execution_time=execution_time
            )
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions"""
        prompt = f"{self.instructions}\n\n"
        
        if self.tools:
            prompt += "Available tools:\n"
            for name, tool in self.tools.items():
                prompt += f"- {name}: {tool['description']}\n"
            
            prompt += "\nTo use a tool, respond with: TOOL_CALL: tool_name(arg1, arg2, ...)\n"
            prompt += "You can use multiple tools in sequence if needed.\n"
            prompt += "Always provide a final response to the user after using tools.\n"
        
        return prompt
    
    async def process_message(self, message: str, max_iterations: int = 5) -> str:
        """Process a user message and return response"""
        # Add user message to history
        self.conversation_history.append(AgentMessage(
            role="user",
            content=message,
            timestamp=datetime.now().isoformat()
        ))
        
        for iteration in range(max_iterations):
            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": self._build_system_prompt()}
            ]
            
            # Add conversation history
            for msg in self.conversation_history[-10:]:  # Keep last 10 messages
                if msg.role in ['user', 'assistant']:
                    messages.append({"role": msg.role, "content": msg.content})
            
            try:
                # Get response from LLM
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                assistant_message = response.choices[0].message.content
                
                # Check if LLM wants to use tools
                tool_calls = self._extract_tool_calls(assistant_message)
                
                if tool_calls:
                    # Execute all tool calls
                    tool_results = []
                    for tool_name, args in tool_calls:
                        result = self._execute_tool(tool_name, args)
                        tool_results.append((tool_name, args, result))
                    
                    # Add assistant message to history
                    self.conversation_history.append(AgentMessage(
                        role="assistant",
                        content=assistant_message,
                        timestamp=datetime.now().isoformat(),
                        metadata={"tool_calls": tool_calls}
                    ))
                    
                    # Add tool results to conversation
                    for tool_name, args, result in tool_results:
                        if result.success:
                            tool_message = f"Tool '{tool_name}' executed successfully. Result: {result.result}"
                        else:
                            tool_message = f"Tool '{tool_name}' failed: {result.error}"
                        
                        self.conversation_history.append(AgentMessage(
                            role="tool",
                            content=tool_message,
                            timestamp=datetime.now().isoformat(),
                            metadata={
                                "tool_name": tool_name,
                                "args": args,
                                "success": result.success,
                                "execution_time": result.execution_time
                            }
                        ))
                    
                    # Continue the loop to get final response
                    continue
                else:
                    # No tool calls, this is the final response
                    self.conversation_history.append(AgentMessage(
                        role="assistant",
                        content=assistant_message,
                        timestamp=datetime.now().isoformat()
                    ))
                    return assistant_message
                    
            except Exception as e:
                error_msg = f"Error in iteration {iteration + 1}: {str(e)}"
                logger.error(error_msg)
                if iteration == max_iterations - 1:
                    return f"I apologize, but I encountered an error: {error_msg}"
        
        return "I reached the maximum number of processing steps. Please try rephrasing your request."
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation"""
        total_messages = len(self.conversation_history)
        tool_calls = sum(1 for msg in self.conversation_history if msg.metadata and 'tool_calls' in msg.metadata)
        
        return {
            "agent_name": self.name,
            "total_messages": total_messages,
            "tool_calls_made": tool_calls,
            "tools_available": list(self.tools.keys()),
            "last_message_time": self.conversation_history[-1].timestamp if self.conversation_history else None
        }

class AgentOrchestrator:
    """Orchestrator for managing multiple agents"""
    
    def __init__(self):
        self.agents: Dict[str, SimpleAgent] = {}
        self.routing_rules: Dict[str, List[str]] = {}  # keyword -> agent_names
        
    def register_agent(self, agent: SimpleAgent, keywords: List[str] = None):
        """Register an agent with optional routing keywords"""
        self.agents[agent.name] = agent
        
        if keywords:
            for keyword in keywords:
                if keyword not in self.routing_rules:
                    self.routing_rules[keyword] = []
                self.routing_rules[keyword].append(agent.name)
        
        logger.info(f"Registered agent '{agent.name}' with keywords: {keywords}")
    
    def route_to_agent(self, message: str) -> str:
        """Route message to the most appropriate agent"""
        message_lower = message.lower()
        
        # Score agents based on keyword matches
        scores = {}
        for keyword, agent_names in self.routing_rules.items():
            if keyword.lower() in message_lower:
                for agent_name in agent_names:
                    scores[agent_name] = scores.get(agent_name, 0) + 1
        
        if scores:
            # Return agent with highest score
            best_agent = max(scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Routed message to '{best_agent}' (score: {scores[best_agent]})")
            return best_agent
        
        # Default to first available agent
        if self.agents:
            default_agent = list(self.agents.keys())[0]
            logger.info(f"No routing match, using default agent: '{default_agent}'")
            return default_agent
        
        raise ValueError("No agents registered")
    
    async def process_message(self, message: str, agent_name: Optional[str] = None) -> Tuple[str, str]:
        """Process message with specified or routed agent"""
        if agent_name is None:
            agent_name = self.route_to_agent(message)
        
        if agent_name not in self.agents:
            return f"Agent '{agent_name}' not found", "error"
        
        agent = self.agents[agent_name]
        response = await agent.process_message(message)
        
        return response, agent_name

# Example tools
def calculate(expression: str) -> str:
    """Calculate mathematical expressions safely"""
    try:
        # Replace common symbols
        safe_expr = expression.replace('^', '**').replace('x', '*')
        # Remove any potentially dangerous characters
        safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', safe_expr)
        
        if not safe_expr.strip():
            return "Invalid expression"
        
        result = eval(safe_expr)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

def get_current_time() -> str:
    """Get the current date and time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_mock_database(query: str) -> str:
    """Search a mock database for information"""
    # Mock database
    database = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
        "openai": "OpenAI is an AI research company that created GPT models and ChatGPT.",
        "machine learning": "Machine learning is a subset of AI that enables computers to learn without explicit programming."
    }
    
    query_lower = query.lower()
    for key, value in database.items():
        if key in query_lower:
            return f"Found information about '{key}': {value}"
    
    return f"No information found for query: '{query}'"

def get_weather(city: str) -> str:
    """Get weather information for a city (mock)"""
    # Mock weather data
    weather_data = {
        "london": "London: 15°C, Cloudy with light rain",
        "paris": "Paris: 18°C, Partly sunny",
        "tokyo": "Tokyo: 25°C, Clear skies",
        "new york": "New York: 12°C, Overcast",
        "sydney": "Sydney: 22°C, Sunny"
    }
    
    city_lower = city.lower()
    return weather_data.get(city_lower, f"{city}: Weather data not available (mock service)")

async def demo_basic_agent():
    """Demo a basic agent with tools"""
    print("🤖 Basic From-Scratch Agent Demo")
    print("=" * 50)
    
    # Create agent
    agent = SimpleAgent(
        name="Assistant",
        instructions="You are a helpful assistant that can perform calculations, tell time, search for information, and get weather updates."
    )
    
    # Add tools
    agent.add_tool("calculate", calculate, "Perform mathematical calculations")
    agent.add_tool("get_time", get_current_time, "Get current date and time")
    agent.add_tool("search", search_mock_database, "Search for information in the database")
    agent.add_tool("weather", get_weather, "Get weather information for a city")
    
    # Test queries
    queries = [
        "What's 15 * 23 + 7?",
        "What time is it?",
        "Tell me about Python programming",
        "What's the weather in Tokyo?",
        "Calculate the area of a circle with radius 5 (use π ≈ 3.14159)"
    ]
    
    for query in queries:
        print(f"\n👤 User: {query}")
        response = await agent.process_message(query)
        print(f"🤖 Agent: {response}")
    
    # Show conversation summary
    summary = agent.get_conversation_summary()
    print(f"\n📊 Conversation Summary: {summary}")

async def demo_multi_agent_orchestrator():
    """Demo multi-agent system with orchestrator"""
    print("\n🔄 Multi-Agent Orchestrator Demo")
    print("=" * 50)
    
    # Create specialized agents
    math_agent = SimpleAgent(
        name="MathBot",
        instructions="You are a mathematics expert. Solve problems step by step and explain your reasoning."
    )
    math_agent.add_tool("calculate", calculate, "Perform mathematical calculations")
    
    info_agent = SimpleAgent(
        name="InfoBot", 
        instructions="You are an information specialist. Search for and provide detailed information on topics."
    )
    info_agent.add_tool("search", search_mock_database, "Search for information")
    info_agent.add_tool("get_time", get_current_time, "Get current time")
    
    weather_agent = SimpleAgent(
        name="WeatherBot",
        instructions="You are a weather specialist. Provide weather information and related advice."
    )
    weather_agent.add_tool("weather", get_weather, "Get weather information")
    
    # Create orchestrator and register agents
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(math_agent, ["calculate", "math", "equation", "solve"])
    orchestrator.register_agent(info_agent, ["search", "information", "tell me", "what is"])
    orchestrator.register_agent(weather_agent, ["weather", "temperature", "forecast"])
    
    # Test queries
    queries = [
        "Calculate 50 * 80 + 15",
        "What's the weather in Paris?",
        "Tell me about artificial intelligence", 
        "What's 2^10?",
        "Weather forecast for London"
    ]
    
    for query in queries:
        print(f"\n👤 User: {query}")
        response, agent_used = await orchestrator.process_message(query)
        print(f"🤖 {agent_used}: {response}")

async def demo_interactive_session():
    """Interactive session with agent"""
    print("\n💬 Interactive Session Demo")
    print("=" * 50)
    print("Type 'quit' to exit, 'summary' for conversation summary")
    
    # Create versatile agent
    agent = SimpleAgent(
        name="Interactive Assistant",
        instructions="You are a helpful interactive assistant. Be conversational and helpful."
    )
    
    # Add all tools
    agent.add_tool("calculate", calculate, "Perform calculations")
    agent.add_tool("get_time", get_current_time, "Get current time")
    agent.add_tool("search", search_mock_database, "Search for information")
    agent.add_tool("weather", get_weather, "Get weather info")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'summary':
                summary = agent.get_conversation_summary()
                print(f"📊 Summary: {summary}")
                continue
            elif not user_input:
                continue
            
            response = await agent.process_message(user_input)
            print(f"🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

async def main():
    """Run all demos"""
    try:
        # Check for API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ Please set OPENAI_API_KEY environment variable")
            print("   export OPENAI_API_KEY=your-api-key-here")
            return
        
        # Run demos
        await demo_basic_agent()
        await demo_multi_agent_orchestrator()
        
        # Ask user if they want interactive session
        choice = input("\n🤔 Would you like to try the interactive session? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            await demo_interactive_session()
        
        print("\n✅ From-scratch agent demos completed!")
        print("\nKey features demonstrated:")
        print("1. Tool execution with error handling")
        print("2. Conversation memory and context")
        print("3. Multi-agent orchestration and routing")
        print("4. Interactive sessions")
        print("5. Monitoring and logging")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Demo failed")

if __name__ == "__main__":
    asyncio.run(main())