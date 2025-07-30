#!/usr/bin/env python3
"""
Basic OpenAI Agent SDK Example
Demonstrates creating a simple agent with tools and handoffs.
"""

import asyncio
import os
from typing import Dict, Any
from agents import Agent, Runner, function_tool

# Set up OpenAI API key
# export OPENAI_API_KEY=your-api-key-here

@function_tool
async def calculate(expression: str) -> str:
    """Calculate mathematical expressions safely.
    
    Args:
        expression: The mathematical expression to evaluate
    """
    try:
        # Simple safe evaluation - extend for production use
        # Replace common math symbols
        safe_expr = expression.replace('^', '**').replace('x', '*')
        result = eval(safe_expr)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

@function_tool
async def get_weather(city: str) -> Dict[str, Any]:
    """Get weather information for a city.
    
    Args:
        city: The name of the city
    """
    # Mock weather data - replace with real API in production
    weather_data = {
        "city": city,
        "temperature": "22°C",
        "condition": "Sunny",
        "humidity": "65%",
        "wind": "10 km/h"
    }
    return weather_data

@function_tool
async def search_web(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: Search query
    """
    # Mock search - replace with real search API
    return f"Here are search results for '{query}': [Mock results - integrate with real search API]"

async def demo_basic_agent():
    """Demo a basic agent with tools"""
    print("🤖 Basic Agent Demo")
    print("=" * 40)
    
    # Create agent with tools
    agent = Agent(
        name="Assistant",
        instructions="""You are a helpful assistant that can:
        - Perform mathematical calculations
        - Get weather information 
        - Search the web for information
        
        Always be friendly and provide clear, helpful responses.""",
        tools=[calculate, get_weather, search_web]
    )
    
    # Test queries
    queries = [
        "What's 15 * 23 + 7?",
        "What's the weather in Tokyo?",
        "Search for information about Python programming",
        "Calculate the square root of 144"
    ]
    
    for query in queries:
        print(f"\n👤 User: {query}")
        result = await Runner.run(agent, query)
        print(f"🤖 Agent: {result.final_output}")

async def demo_multi_agent_system():
    """Demo a multi-agent system with handoffs"""
    print("\n🔄 Multi-Agent System Demo")
    print("=" * 40)
    
    # Specialized agents
    math_agent = Agent(
        name="Math Specialist",
        handoff_description="Expert in mathematical calculations and problem solving",
        instructions="""You are a mathematics expert. 
        - Solve problems step by step
        - Explain your reasoning clearly
        - Handle complex mathematical operations""",
        tools=[calculate]
    )
    
    weather_agent = Agent(
        name="Weather Specialist", 
        handoff_description="Expert in weather information and forecasting",
        instructions="""You provide weather information and forecasts.
        - Be detailed and helpful
        - Provide context about weather conditions
        - Suggest appropriate activities based on weather""",
        tools=[get_weather]
    )
    
    research_agent = Agent(
        name="Research Specialist",
        handoff_description="Expert in finding and summarizing information",
        instructions="""You help users find information by searching and summarizing.
        - Provide comprehensive answers
        - Cite sources when possible
        - Organize information clearly""",
        tools=[search_web]
    )
    
    # Coordinator agent
    coordinator = Agent(
        name="Coordinator",
        instructions="""You coordinate between specialists based on user requests:
        - Route math questions to Math Specialist
        - Route weather questions to Weather Specialist  
        - Route research/information questions to Research Specialist
        - For complex requests, you may need to coordinate between multiple specialists""",
        handoffs=[math_agent, weather_agent, research_agent]
    )
    
    # Test complex queries
    complex_queries = [
        "Calculate 50 * 80 + 15, then tell me the weather in London",
        "What's the weather in Paris and find information about French cuisine?",
        "I need to calculate my monthly budget (salary $5000, expenses $3200) and find weather info for my vacation to Miami"
    ]
    
    for query in complex_queries:
        print(f"\n👤 User: {query}")
        result = await Runner.run(coordinator, query)
        print(f"🤖 Coordinator: {result.final_output}")

async def demo_with_memory():
    """Demo agent with conversation memory"""
    print("\n🧠 Memory Demo")
    print("=" * 40)
    
    agent = Agent(
        name="Memory Assistant",
        instructions="""You are a helpful assistant with memory.
        Remember context from previous conversations and reference it when relevant.""",
        tools=[calculate, get_weather]
    )
    
    # Conversation with memory
    conversation = [
        "My name is Alice and I live in San Francisco",
        "What's the weather like in my city?", 
        "Can you calculate 25% of 1000 for my budget?",
        "Thanks! What was my name again?"
    ]
    
    for message in conversation:
        print(f"\n👤 Alice: {message}")
        result = await Runner.run(agent, message)
        print(f"🤖 Agent: {result.final_output}")

async def main():
    """Run all demos"""
    try:
        # Check for API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ Please set OPENAI_API_KEY environment variable")
            print("   export OPENAI_API_KEY=your-api-key-here")
            return
        
        await demo_basic_agent()
        await demo_multi_agent_system()
        await demo_with_memory()
        
        print("\n✅ All demos completed!")
        print("\nNext steps:")
        print("1. Try modifying the agent instructions")
        print("2. Add your own custom tools") 
        print("3. Experiment with different handoff patterns")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have installed: pip install openai-agents")

if __name__ == "__main__":
    asyncio.run(main())