#!/usr/bin/env python3
"""
Test OpenAI MCP Client with .env configuration
Tests connection to Pandas MCP Server on Huawei ECS
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class TestOpenAIMCPClient:
    def __init__(self):
        # Load configuration from environment
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.mcp_url = os.getenv("MCP_SSE_URL")
        
        # Validate configuration
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file")
        
        print("📋 Configuration loaded from .env:")
        print(f"   - Model: {self.model}")
        print(f"   - MCP Server: {self.mcp_url[:10]}")
        print(f"   - API Key: {self.api_key[:3]}...")
        
        # Initialize OpenAI client
        self.openai = AsyncOpenAI(api_key=self.api_key)
        
        # Session management
        self.session: Optional[ClientSession] = None
        self.messages: List[Dict[str, Any]] = []
        self.available_tools = []

    async def connect(self):
        """Connect to MCP server"""
        print(f"\n🔌 Connecting to MCP server at {self.mcp_url}...")
        
        try:
            self._streams_context = sse_client(url=self.mcp_url)
            streams = await self._streams_context.__aenter__()

            self._session_context = ClientSession(*streams)
            self.session = await self._session_context.__aenter__()

            await self.session.initialize()

            # Get available tools
            response = await self.session.list_tools()
            tools = response.tools
            
            # Convert to OpenAI format
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
            
            print(f"✅ Connected successfully!")
            print(f"📦 Found {len(tools)} tools")
            
            # List first 5 tools
            print("\n🔧 Available tools (first 5):")
            for tool in tools[:5]:
                print(f"   - {tool.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\nTroubleshooting:")
            print("1. Check if MCP_SSE_URL in .env is correct")
            print("2. Ensure port 8000 is open in Huawei security group")
            print("3. Verify server is running: ssh to ECS and run 'docker ps'")
            return False

    async def cleanup(self):
        """Clean up connections"""
        if hasattr(self, '_session_context'):
            await self._session_context.__aexit__(None, None, None)
        if hasattr(self, '_streams_context'):
            await self._streams_context.__aexit__(None, None, None)

    async def test_server_info(self):
        """Test getting server information"""
        print("\n🧪 Test 1: Get server information")
        print("-" * 40)
        
        try:
            result = await self.session.call_tool("get_server_info_tool", {})
            
            # Parse result
            if hasattr(result, 'content'):
                content = result.content
                if hasattr(content, '__iter__'):  # It's a list
                    for item in content:
                        if hasattr(item, 'text'):
                            info = json.loads(item.text)
                            break
                elif hasattr(content, 'text'):
                    info = json.loads(content.text)
                else:
                    info = content
            else:
                info = result
            
            print("✅ Server info retrieved:")
            print(f"   - Version: {info.get('server', {}).get('version', 'Unknown')}")
            print(f"   - Name: {info.get('server', {}).get('name', 'Unknown')}")
            print(f"   - Transport: {info.get('server', {}).get('transport', 'Unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False

    async def test_list_dataframes(self):
        """Test listing DataFrames"""
        print("\n🧪 Test 2: List DataFrames")
        print("-" * 40)
        
        try:
            result = await self.session.call_tool("list_dataframes_tool", {})
            print("✅ DataFrames listed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False

    async def test_openai_integration(self):
        """Test OpenAI integration with tools"""
        print("\n🧪 Test 3: OpenAI Integration")
        print("-" * 40)
        
        query = "What pandas analysis tools are available?"
        print(f"Query: {query}")
        
        response = await self.process_query(query)
        print(f"Response: {response[:300]}..." if len(response) > 300 else f"Response: {response}")
        
        return True

    async def test_create_and_load_data(self):
        """Test creating and loading sample data"""
        print("\n🧪 Test 4: Create and Load Sample Data")
        print("-" * 40)
        
        query = "Create a simple CSV file with 3 rows of sample sales data and tell me what you created"
        print(f"Query: {query}")
        
        response = await self.process_query(query)
        print(f"Response: {response[:500]}..." if len(response) > 500 else f"Response: {response}")
        
        return True
    
    async def test_upload_and_load_flow(self):
        """Test that upload and load work together"""
        print("\n🧪 Test 5: Upload and Load Flow")
        print("-" * 40)
        
        try:
            # Upload a file
            upload_result = await self.session.call_tool("upload_temp_file_tool", {
                "filename": "test.csv",
                "content": "name,value\nA,1\nB,2",
                "session_id": "test"
            })
            
            # Parse upload result
            if hasattr(upload_result, 'content'):
                content = upload_result.content
                if hasattr(content, '__iter__'): 
                    for item in content:
                        if hasattr(item, 'text'):
                            upload_data = json.loads(item.text)
                            break
                elif hasattr(content, 'text'):
                    upload_data = json.loads(content.text)
                else:
                    upload_data = content
            else:
                upload_data = upload_result
            
            # Check upload success
            if not upload_data.get('success'):
                print(f"❌ Upload failed: {upload_data.get('error')}")
                return False
            
            print(f"✅ File uploaded: {upload_data.get('filepath')}")
            
            # Load using the returned filepath
            load_result = await self.session.call_tool("load_dataframe_tool", {
                "filepath": upload_data['filepath'],
                "df_name": "test_df",
                "session_id": "test"
            })
            
            # Parse load result
            if hasattr(load_result, 'content'):
                content = load_result.content
                if hasattr(content, '__iter__'):  
                    for item in content:
                        if hasattr(item, 'text'):
                            load_data = json.loads(item.text)
                            break
                elif hasattr(content, 'text'):
                    load_data = json.loads(content.text)
                else:
                    load_data = content
            else:
                load_data = load_result
            
            if load_data.get('success'):
                print(f"✅ DataFrame loaded: {load_data.get('dataframe_info', {}).get('name')}")
                return True
            else:
                print(f"❌ Load failed: {load_data.get('error')}")
                return False
                
        except Exception as e:
            print(f"❌ Upload/Load flow failed: {e}")
            return False

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and MCP tools"""
        
        self.messages.append({"role": "user", "content": query})
        
        try:
            # Call OpenAI with tools
            response = await self.openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.available_tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_message = response.choices[0].message
            
            # Handle tool calls if any
            if assistant_message.tool_calls:
                # Store assistant message
                tool_call_msg = {
                    "role": "assistant",
                    "content": assistant_message.content
                }
                
                if assistant_message.tool_calls:
                    tool_call_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in assistant_message.tool_calls
                    ]
                
                self.messages.append(tool_call_msg)
                
                # Execute tool calls
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"   🔧 Calling tool: {tool_name}")
                    
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        
                        # Parse result
                        if hasattr(result, 'content'):
                            if isinstance(result.content, list):
                                tool_result = ""
                                for item in result.content:
                                    if hasattr(item, 'text'):
                                        tool_result += item.text
                                    else:
                                        tool_result += str(item)
                            elif hasattr(result.content, 'text'):
                                tool_result = result.content.text
                            else:
                                tool_result = str(result.content)
                        else:
                            tool_result = str(result)
                        
                        # Try to parse JSON if possible
                        try:
                            parsed = json.loads(tool_result)
                            tool_result = json.dumps(parsed, indent=2)
                        except:
                            pass
                        
                        # Limit size
                        if len(tool_result) > 2000:
                            tool_result = tool_result[:2000] + "\n...(truncated)"
                        
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                        
                    except Exception as e:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {str(e)}"
                        })
                
                # Get final response
                final_response = await self.openai.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                final_message = final_response.choices[0].message
                self.messages.append({
                    "role": "assistant",
                    "content": final_message.content
                })
                
                return final_message.content
            
            else:
                # No tool calls
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })
                return assistant_message.content
                
        except Exception as e:
            return f"Error: {str(e)}"

    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print(" 🧪 RUNNING ALL TESTS")
        print("="*60)
        
        tests_passed = 0
        tests_failed = 0
        
        # Test 1: Server info
        if await self.test_server_info():
            tests_passed += 1
        else:
            tests_failed += 1
        
        # Test 2: List DataFrames
        if await self.test_list_dataframes():
            tests_passed += 1
        else:
            tests_failed += 1
        
        # Test 3: OpenAI integration
        if await self.test_openai_integration():
            tests_passed += 1
        else:
            tests_failed += 1
        
        # Test 4: Create and load data

        if await self.test_create_and_load_data():
            tests_passed += 1
        else:
            tests_failed += 1
        
        # Test 5: Upload and load flow

        if await self.test_upload_and_load_flow():
            tests_passed += 1
        else:
            tests_failed +=1

        # Summary
        print("\n" + "="*60)
        print(" 📊 TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed: {tests_passed}")
        print(f"❌ Failed: {tests_failed}")
        print(f"📈 Success Rate: {(tests_passed/(tests_passed+tests_failed)*100):.1f}%")
        
        return tests_failed == 0


async def main():
    """Main test function"""
    print("🚀 OpenAI + MCP Integration Test Suite")
    print("="*60)
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env file")
        print("\nPlease ensure your .env file contains:")
        print("OPENAI_API_KEY=sk-...")
        print("OPENAI_MODEL=gpt-4o-mini")
        print("MCP_SSE_URL=http://119.13.110.147:8000/sse")
        return
    
    if not os.getenv("MCP_SSE_URL"):
        print("⚠️  MCP_SSE_URL not found in .env, using default")
        print("   Default: http://119.13.110.147:8000/sse")
    
    # Create test client
    client = TestOpenAIMCPClient()
    
    try:
        # Connect to server
        if not await client.connect():
            print("\n❌ Failed to connect to MCP server")
            print("Please check:")
            print("1. Server is running on ECS")
            print("2. Port 8000 is open in security group")
            print("3. MCP_SSE_URL in .env is correct")
            return
        
        # Run all tests
        all_passed = await client.run_all_tests()
        
        if all_passed:
            print("\n✅ All tests passed!")
        else:
            print("\n⚠️  Some tests failed")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.cleanup()
        print("\n🧹 Cleanup complete")


if __name__ == "__main__":
    # Check for required packages
    try:
        import mcp
        import openai
        from dotenv import load_dotenv
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\nInstall with:")
        print("pip install mcp openai python-dotenv")
        sys.exit(1)
    
    # Run tests
    asyncio.run(main())