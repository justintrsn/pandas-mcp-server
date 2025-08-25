#!/usr/bin/env python3
"""
MCP Client using OpenAI API with official MCP client library
Connects to Pandas MCP Server via SSE and uses OpenAI for intelligent interaction
"""

import asyncio
import json
import os
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class OpenAIMCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.openai = AsyncOpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        # Store conversation history
        self.messages: List[Dict[str, Any]] = []
        
        # Store available tools for reference
        self.available_tools = []

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        print(f"Connecting to MCP server at {server_url}...")
        
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client successfully!")
        print("\nDiscovering available tools...")
        response = await self.session.list_tools()
        tools = response.tools
        
        # Convert MCP tools to OpenAI function format
        self.available_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"Tool: {tool.name}",
                    "parameters": tool.inputSchema if tool.inputSchema else {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            self.available_tools.append(openai_tool)
        
        print(f"\n✅ Connected to Pandas MCP Server")
        print(f"📦 Available tools: {len(tools)}")
        
        # Show first few tools
        for i, tool in enumerate(tools[:5]):
            print(f"  - {tool.name}: {(tool.description or '')[:60]}...")
        
        if len(tools) > 5:
            print(f"  ... and {len(tools) - 5} more tools")

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if hasattr(self, '_session_context') and self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if hasattr(self, '_streams_context') and self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    def _format_tools_list(self) -> str:
        """Format the list of available tools for display"""
        if not self.available_tools:
            return "No tools available"
        
        tools_list = []
        for tool in self.available_tools:
            func = tool.get("function", {})
            name = func.get("name", "Unknown")
            desc = func.get("description", "No description")
            params = func.get("parameters", {})
            
            # Format parameters
            props = params.get("properties", {})
            required = params.get("required", [])
            
            param_list = []
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                is_required = param_name in required
                req_str = " (required)" if is_required else " (optional)"
                param_list.append(f"    - {param_name} ({param_type}){req_str}: {param_desc}")
            
            tool_info = f"• {name}\n  {desc}"
            if param_list:
                tool_info += "\n  Parameters:\n" + "\n".join(param_list)
            
            tools_list.append(tool_info)
        
        return "\n\n".join(tools_list)

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available MCP tools"""
        
        # Check if user is asking about available tools
        if any(phrase in query.lower() for phrase in ['what tools', 'list tools', 'available tools', 'all tools']):
            # Provide a formatted list of tools
            tools_info = self._format_tools_list()
            # Add this as context for the AI
            system_message = f"The user is asking about available tools. Here's the complete list:\n\n{tools_info}\n\nProvide a well-formatted response about these tools."
            
            # Add system context temporarily
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Add previous conversation context
            messages.extend(self.messages)
            messages.append({"role": "user", "content": query})
            
            # Get response without tools
            response = await self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            assistant_message = response.choices[0].message
            self.messages.append({
                "role": "user",
                "content": query
            })
            self.messages.append({
                "role": "assistant",
                "content": assistant_message.content
            })
            
            return assistant_message.content
        
        # Regular query processing
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": query
        })
        
        try:
            # Call OpenAI with tools
            response = await self.openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.available_tools,
                tool_choice="auto",  # Let the model decide when to use tools
                temperature=0.7,
                max_tokens=2000
            )
            
            # Get the assistant's response
            assistant_message = response.choices[0].message
            
            # Check if the model wants to use tools
            if assistant_message.tool_calls:
                # Add assistant's message with tool calls to history
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content if assistant_message.content else None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"\n🔧 Calling tool: {tool_name}")
                    print(f"   Args: {json.dumps(tool_args, indent=2)}")
                    
                    try:
                        # Execute tool call via MCP
                        result = await self.session.call_tool(tool_name, tool_args)
                        
                        # Format result - handle MCP response objects
                        if hasattr(result, 'content'):
                            # Handle TextContent or other content objects
                            if hasattr(result.content, 'text'):
                                tool_result = result.content.text
                            elif isinstance(result.content, list):
                                # Handle list of content items
                                tool_result = []
                                for item in result.content:
                                    if hasattr(item, 'text'):
                                        tool_result.append(item.text)
                                    else:
                                        tool_result.append(str(item))
                                tool_result = '\n'.join(tool_result) if tool_result else str(result.content)
                            else:
                                # Try to get string representation
                                tool_result = str(result.content)
                        else:
                            tool_result = str(result)
                        
                        # Parse JSON strings if possible
                        if isinstance(tool_result, str):
                            try:
                                # Try to parse as JSON
                                parsed = json.loads(tool_result)
                                tool_result = parsed
                            except (json.JSONDecodeError, ValueError):
                                # Not JSON, keep as string
                                pass
                        
                        # Truncate very long results
                        if isinstance(tool_result, str) and len(tool_result) > 5000:
                            tool_result = tool_result[:5000] + "\n... (truncated)"
                        
                        # Serialize for OpenAI
                        if isinstance(tool_result, (dict, list)):
                            content = json.dumps(tool_result, indent=2)
                        else:
                            content = str(tool_result)
                        
                        # Add tool result to messages
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content
                        })
                        
                    except Exception as e:
                        error_msg = f"Tool execution failed: {str(e)}"
                        print(f"   ❌ {error_msg}")
                        
                        # Add error as tool result
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg
                        })
                
                # Get final response from OpenAI with tool results
                final_response = await self.openai.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                final_message = final_response.choices[0].message
                self.messages.append({
                    "role": "assistant",
                    "content": final_message.content
                })
                
                return final_message.content
            
            else:
                # No tool calls, just return the response
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })
                return assistant_message.content
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"\n❌ {error_msg}")
            return error_msg

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\n" + "="*60)
        print(" OPENAI + MCP PANDAS SERVER - INTERACTIVE CHAT")
        print("="*60)
        print("\nYou can now interact with your data using natural language!")
        print("The AI has access to all pandas operations and visualization tools.")
        print("\nExample queries:")
        print("  - 'Load the file sales_data.csv'")
        print("  - 'Show me the first 5 rows of the data'")
        print("  - 'What is the average sales by category?'")
        print("  - 'Create a bar chart of revenue by month'")
        print("  - 'Analyze the correlation between price and quantity'")
        print("\nType 'quit' to exit, 'clear' to reset conversation")
        print("-"*60)
        
        while True:
            try:
                # Get user input
                query = input("\n🤖 You: ").strip()
                
                if query.lower() == 'quit':
                    print("\nGoodbye! 👋")
                    break
                
                if query.lower() == 'clear':
                    self.messages = []
                    print("✅ Conversation cleared")
                    continue
                
                if query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  quit  - Exit the chat")
                    print("  clear - Clear conversation history")
                    print("  help  - Show this help message")
                    print("\nYou can ask me to:")
                    print("  - Load and analyze data files")
                    print("  - Run pandas operations")
                    print("  - Create visualizations")
                    print("  - Get insights from your data")
                    continue
                
                if not query:
                    continue
                
                # Process the query
                print("\n🤔 Thinking...")
                response = await self.process_query(query)
                
                # Display response
                print(f"\n💬 Assistant: {response}")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'quit' to exit.")

    async def run_example_session(self):
        """Run an example session to demonstrate capabilities"""
        print("\n" + "="*60)
        print(" RUNNING EXAMPLE SESSION")
        print("="*60)
        
        # Create sample data
        import pandas as pd
        sample_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'sales': [100 + i*5 + (i%7)*10 for i in range(30)],
            'category': ['Electronics', 'Clothing', 'Food'] * 10,
            'profit': [20 + i*2 for i in range(30)]
        })
        sample_df.to_csv('sample_sales.csv', index=False)
        print("\n✅ Created sample_sales.csv for demonstration")
        
        # Example queries
        example_queries = [
            "Load the file sample_sales.csv as 'sales' DataFrame",
            "Show me basic statistics about the sales data",
            "What is the total sales by category?",
            "Create a bar chart showing sales by category"
        ]
        
        print("\nRunning example queries...")
        print("-"*40)
        
        for query in example_queries:
            print(f"\n📝 Query: {query}")
            response = await self.process_query(query)
            print(f"📊 Response: {response[:500]}..." if len(response) > 500 else f"📊 Response: {response}")
            await asyncio.sleep(1)  # Small delay between queries
        
        print("\n" + "="*60)
        print(" Example session complete!")
        print(" Now entering interactive mode...")
        print("="*60)


async def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr create a .env file with:")
        print("  OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
    
    # Parse arguments properly
    server_url = "http://localhost:8000/sse"  # Default
    run_example = False
    
    # Parse command line arguments
    args = sys.argv[1:]  # Skip script name
    for arg in args:
        if arg == "--example":
            run_example = True
        elif not arg.startswith("--"):  # If it's not a flag, treat as URL
            server_url = arg
    
    if not server_url.startswith(("http://", "https://")):
        print(f"⚠️  Server URL should start with http:// or https://")
        server_url = f"http://{server_url}"
    
    print(f"Using server URL: {server_url}")
    
    # Create and run client
    client = OpenAIMCPClient()
    
    try:
        # Connect to server
        await client.connect_to_sse_server(server_url)
        
        # Optional: Run example session first
        if run_example:
            await client.run_example_session()
        
        # Start interactive chat
        await client.chat_loop()
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await client.cleanup()
        print("\n✅ Cleanup complete")


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import mcp
        from openai import AsyncOpenAI
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp", "openai", "python-dotenv"])
        print("✅ Packages installed. Please run the script again.")
        sys.exit(0)
    
    # Run the main function
    asyncio.run(main())