
# OpenAI MCP Client for Pandas Server - User Guide


## 📋 Table of Contents

-   Overview
-   Prerequisites
-   Installation
-   Setup
-   Running the System
-   Using the Client
-   Example Workflows
-   Troubleshooting

## Overview

The OpenAI MCP Client provides an intelligent natural language interface to the Pandas MCP Server. It combines OpenAI's language models with the MCP protocol to enable conversational data analysis, allowing you to:

-   Load and analyze data files using natural language
-   Execute pandas operations through conversation
-   Create visualizations with simple requests
-   Get insights and summaries from your data
-   All without writing code!

## Prerequisites

### System Requirements

-   Python 3.10 or higher
-   macOS, Linux, or Windows (with WSL recommended)
-   At least 4GB RAM
-   Internet connection (for OpenAI API)

### Required Accounts

-   **OpenAI API Key**: Sign up at [platform.openai.com](https://platform.openai.com/) and create an API key

## Installation

### Step 1: Install UV Package Manager

`UV` is a fast Python package installer and resolver. `UV` is preffered due to performance and it is recommended to be used in conjunction with `fastmcp` as it is the current best practice (although pip is still perfectly fine). Install it using one of these methods:

**macOS/Linux:**

```bash
# Using curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

```

**Windows:**

```powershell
# Using PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv

```

After installation, restart your terminal and verify:

```bash
uv --version

```

### Step 2: Create Virtual Environment with UV

```bash
# Create a new virtual environment
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

```

### Step 3: Install Dependencies

```bash
# Install all dependencies using UV (faster than pip)
uv pip install -r requirements.txt

# Install additional client dependencies
uv pip install openai mcp python-dotenv

```

## Setup

### Step 1: Configure OpenAI API Key

Create a `.env` file in the project root:

```bash
touch .env

```

Add your OpenAI API key to the `.env` file:

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4-turbo-preview  # Optional: specify model (default: gpt-4-turbo-preview)

```

**Security Note:** Never commit your `.env` file to version control!

### Step 2: Verify Installation

Run the test suite to ensure everything is working:

```bash
python test/test.py

```

You should see all tests passing:

```
✅ ALL TESTS PASSED!
The server is ready to use

```

## Running the System

### Step 1: Start the Pandas MCP Server

In one terminal window, start the MCP server:

```bash
# Activate virtual environment if not already active
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Start the server
python server.py

```

You should see:

```
============================================================
 pandas-mcp-server v0.1.0
 A powerful MCP server for pandas data analysis with Chart.js visualizations
============================================================

Server running at http://0.0.0.0:8000
Connect your MCP client to this URL

```

Keep this terminal running!

### Step 2: Start the OpenAI Client

In a **new terminal window**:

```bash
# Navigate to the project directory
cd pandas-mcp-server

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Start the client (connects to default localhost:8000)
python client.py

# Or specify a custom server URL
python client.py http://localhost:8000/sse

```

You should see:

```
Connecting to MCP server at http://localhost:8000/sse...
Initialized SSE client successfully!

✅ Connected to Pandas MCP Server
📦 Available tools: 17
  - read_metadata_tool: Extract comprehensive metadata from data files...
  - run_pandas_code_tool: Execute pandas operations on DataFrames...
  ... and more tools

============================================================
 OPENAI + MCP PANDAS SERVER - INTERACTIVE CHAT
============================================================

You can now interact with your data using natural language!

```

## Using the Client

### Basic Commands

The client provides an interactive chat interface. Here are the basic commands:

Command

Description

`help`

Show available commands and capabilities

`clear`

Clear conversation history

`quit`

Exit the client

### Natural Language Queries

You can ask the AI to perform any data operation using natural language:

#### Loading Data

```
You: Load the file sales_data.csv

You: Read the Excel file reports.xlsx and name it 'reports'

You: Load data.json as a DataFrame called 'json_data'

```

#### Data Exploration

```
You: Show me the first 5 rows of the data

You: What are the column names and data types?

You: Give me summary statistics for all numeric columns

You: How many missing values are in each column?

```

#### Data Analysis

```
You: What is the average sales by category?

You: Group the data by month and calculate total revenue

You: Find the correlation between price and quantity

You: Show me the top 10 products by revenue

```

#### Visualizations

```
You: Create a bar chart showing sales by category

You: Make a line chart of revenue over time

You: Generate a correlation heatmap for all numeric columns

You: Create a pie chart of market share by product

```

### Viewing Available Tools

To see all available MCP tools:

```
You: What tools are available?

You: List all tools with their parameters

```

## Example Workflows

### Workflow 1: Sales Data Analysis

```bash
# Start with sample data
You: Load the file sample_sales.csv as 'sales'

# Explore the data
You: Show me the structure of the sales data

# Analyze
You: What are the total sales by category?

# Visualize
You: Create a bar chart comparing sales across categories

# Deep dive
You: Show me the trend of sales over time for Electronics only

```

### Workflow 2: Correlation Analysis

```bash
# Load data
You: Load features.csv

# Check correlations
You: Calculate the correlation matrix for all numeric columns

# Visualize
You: Create a correlation heatmap

# Identify relationships
You: Which features have the strongest correlation with the target variable?

```

### Workflow 3: Data Quality Check

```bash
# Load and inspect
You: Load customer_data.xlsx

# Quality check
You: Check for missing values and duplicates

# Get metadata
You: Extract comprehensive metadata about this file

# Clean data
You: Remove rows with missing values in the 'email' column

```

### Running the Example Session

The client includes a built-in example session:

```bash
python client.py --example

```

This will:

1.  Create a sample dataset
2.  Run example queries
3.  Demonstrate various capabilities
4.  Then enter interactive mode

## Advanced Features

### Custom Server Configuration

If your server is running on a different host or port:

```bash
# Connect to remote server
python client.py http://your-server:8000/sse

# Connect with custom configuration
export MCP_SERVER_HOST=192.168.1.100
export MCP_SERVER_PORT=9000
python client.py http://192.168.1.100:9000/sse

```

### Model Selection

You can choose different OpenAI models in your `.env` file:

```env
# For faster responses (less capable)
OPENAI_MODEL=gpt-3.5-turbo

# For best quality (default)
OPENAI_MODEL=gpt-4-turbo-preview

# For latest GPT-4
OPENAI_MODEL=gpt-4

```

### Session Management

The client maintains conversation context:

```bash
# Continue a complex analysis
You: Load sales.csv
You: Calculate monthly totals
You: Now show me the year-over-year growth  # Remembers previous context

# Clear when starting new analysis
You: clear
You: Load different_data.csv  # Fresh context

```

## Data File Management

### Supported File Formats

The system supports these file types:

-   **CSV** (.csv) - Comma-separated values
-   **TSV** (.tsv) - Tab-separated values
-   **Excel** (.xlsx, .xls) - Microsoft Excel
-   **JSON** (.json) - JavaScript Object Notation
-   **Parquet** (.parquet) - Apache Parquet format

### File Location

Place your data files in the project directory or subdirectories:

```
pandas-mcp-server/
├── data/           # Recommended data directory
│   ├── sales.csv
│   ├── reports.xlsx
│   └── analysis.json
├── test/           # Test data
└── your_file.csv   # Or in root directory

```

Reference files using relative paths:

```
You: Load data/sales.csv
You: Load test/sample.xlsx

```

## Troubleshooting

### Common Issues and Solutions

#### 1. OpenAI API Key Error

```
❌ Error: OPENAI_API_KEY environment variable not set

```

**Solution:** Create `.env` file with your API key or export it:

```bash
export OPENAI_API_KEY='sk-your-key-here'

```

#### 2. Connection Failed

```
Failed to connect to MCP server

```

**Solution:**

-   Ensure the server is running in another terminal
-   Check the server URL is correct
-   Verify firewall settings

#### 3. Module Not Found

```
ModuleNotFoundError: No module named 'mcp'

```

**Solution:** Install missing dependencies:

```bash
uv pip install mcp openai python-dotenv

```

#### 4. Server Port Already in Use

```
Address already in use

```

**Solution:** Use a different port:

```bash
export MCP_SERVER_PORT=8001
python server.py
# Then connect client to new port
python client.py http://localhost:8001/sse

```

#### 5. Rate Limit Errors

```
Rate limit exceeded for OpenAI API

```

**Solution:**

-   Wait a moment before retrying
-   Consider upgrading your OpenAI plan
-   Use a less expensive model like gpt-3.5-turbo

### Getting Help

1.  **Check logs:** Server logs are in `logs/pandas_mcp.log`
2.  **Verbose mode:** Set `LOG_LEVEL=DEBUG` in environment
3.  **Test individual components:**
    
    ```bash
    python test/test_metadata.pypython test/test_visualization.py
    
    ```
    

## Best Practices

### 1. Data Preparation

-   Keep data files under 100MB for best performance
-   Use meaningful column names
-   Ensure consistent data types in columns

### 2. Query Formulation

-   Be specific about what you want
-   Reference DataFrame names when working with multiple datasets
-   Break complex analyses into steps

### 3. Session Management

-   Use `clear` between unrelated analyses
-   Save important results to files
-   Keep conversation focused on one dataset at a time

### 4. Performance Tips

-   For large datasets, use sampling: "Analyze a sample of 1000 rows"
-   Create indexes on frequently queried columns
-   Use appropriate aggregations before visualization

## Examples Gallery

### Quick Data Profiling

```
You: Load customers.csv and give me a complete data profile including statistics, data types, missing values, and recommendations

```

### Multi-Step Analysis

```
You: Load sales_2024.csv
You: Filter for Q4 data only
You: Group by product category and region
You: Calculate average order value for each group
You: Create a heatmap showing the results

```

### Automated Reporting

```
You: Load monthly_data.xlsx
You: Generate a summary report with:
     1. Key statistics
     2. Month-over-month growth
     3. Top 5 performers
     4. A trend chart

```

## Security Notes

1.  **API Keys:** Never share or commit your OpenAI API key
2.  **Data Privacy:** The OpenAI API processes your queries but not your actual data
3.  **File Access:** The server only accesses files you explicitly load
4.  **Code Execution:** All pandas operations are sandboxed for security

## Available MCP Tools

The client has access to these powerful tools:

### Data Loading & Management

-   `load_dataframe_tool` - Load CSV, Excel, JSON, Parquet files
-   `list_dataframes_tool` - List all loaded DataFrames
-   `get_dataframe_info_tool` - Get detailed DataFrame information
-   `preview_file_tool` - Preview files without loading

### Data Analysis

-   `read_metadata_tool` - Extract comprehensive file metadata
-   `run_pandas_code_tool` - Execute pandas operations
-   `validate_pandas_code_tool` - Validate code before execution
-   `get_execution_context_tool` - Get execution environment info

### Visualization

-   `create_chart_tool` - Create interactive charts (bar, line, pie, scatter)
-   `suggest_charts_tool` - Get chart recommendations based on data
-   `create_correlation_heatmap_tool` - Generate correlation matrices
-   `create_time_series_chart_tool` - Specialized time series plots
-   `get_chart_types_tool` - List all visualization options

### Session Management

-   `clear_session_tool` - Clear session data
-   `get_session_info_tool` - Get session information
-   `get_server_info_tool` - Get server configuration

## Tips for Effective Usage

### 1. Start Simple

Begin with basic queries and gradually build complexity:

```
You: Load data.csv
You: Show me the shape
You: What columns are available?
You: Show summary statistics

```

### 2. Use Descriptive Names

When loading multiple datasets, use clear names:

```
You: Load sales_2023.csv as 'last_year'
You: Load sales_2024.csv as 'this_year'
You: Compare last_year and this_year revenue

```

### 3. Leverage Context

The AI remembers your conversation:

```
You: Filter for products with price > 100
You: Now group these by category  # AI knows to use filtered data
You: Create a pie chart of the results  # Uses the grouped data

```

### 4. Ask for Explanations

The AI can explain its analysis:

```
You: Explain what the correlation matrix shows
You: What insights can we draw from this chart?
You: What data quality issues should I address?

```

## Next Steps

1.  **Explore Visualizations:** Try different chart types with your data
2.  **Complex Analysis:** Combine multiple tools for sophisticated analysis
3.  **Automation:** Save common workflows as conversation templates
4.  **Integration:** Use the MCP server with other MCP-compatible clients

## Support

For issues or questions:

1.  Check the troubleshooting section
2.  Review test outputs: `python test/test.py`
3.  Examine server logs: `logs/pandas_mcp.log`
4.  Go ask Woo Yan Kit 84298208 to debug your issue (please do not bother Justin Trisno 84400747 as he won't have time)
5.  Ask chatgpt or any LLM provider for help
6.  God speed!

----------

**Happy Data Analysis! 🎉**
