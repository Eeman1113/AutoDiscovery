# Autodisc - Multi-Agent Autonomous Research and Discovery System

A sophisticated autonomous research platform that transforms user queries into comprehensive, structured documentation through coordinated agent collaboration with minimal human intervention.

## 🌟 Features

- **Multi-Agent Architecture**: Six specialized agents working in coordination
- **Autonomous Operation**: Fully automated research workflow
- **Comprehensive Analysis**: Deep pattern recognition and creative problem-solving
- **Quality Assurance**: Multi-layer validation and fact-checking
- **Multiple Output Formats**: Markdown, HTML, JSON, and PDF support
- **Interactive CLI**: Command-line interface with interactive mode
- **Extensible Design**: Plugin architecture for easy customization

## 🏗️ System Architecture

### Agent Overview

1. **Coordinator Agent** - Master controller orchestrating the entire workflow
2. **Research Agent** - Comprehensive information gathering and source identification
3. **Analysis Agent** - Deep analysis, pattern recognition, and information synthesis
4. **Creative Agent** - Novel solution generation and creative problem-solving
5. **Validation Agent** - Quality assurance, fact-checking, and verification
6. **Documentation Agent** - Structured documentation creation and formatting

### Workflow Process

1. **Query Processing** - Parse and analyze user input
2. **Strategy Formation** - Determine optimal research approach
3. **Parallel Research** - Multiple agents work simultaneously
4. **Synthesis & Integration** - Cross-agent collaboration and refinement
5. **Quality Assurance** - Comprehensive validation and fact-checking
6. **Documentation** - Final structured output generation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Autodisc
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

4. **Run the example**
   ```bash
   python main.py
   ```

### Basic Usage

#### Command Line Interface

```bash
# Execute a research query
python main.py --query "What are the economic implications of vertical farming in urban environments?"

# Specify domain and depth
python main.py --query "Design a new natural language with 5 letters" --domain "linguistics" --depth "comprehensive"

# Save results to file
python main.py --query "Analyze the impact of AI on healthcare" --output results.json

# Interactive mode
python main.py --interactive
```

#### Interactive Mode

```bash
python main.py --interactive
```

Available commands:
- `query: <your research question>` - Execute a research query
- `status` - Show system status
- `sessions` - List active sessions
- `help` - Show available commands
- `quit` - Exit interactive mode

#### Example Queries

```bash
# Creative query
query: Design a new natural language with 5 letters

# Analytical query
query: What are the economic implications of vertical farming in urban environments?

# Technical query
query: Analyze the impact of artificial intelligence on healthcare delivery

# Business query
query: Evaluate the market potential for sustainable packaging solutions
```

## 📋 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### System Configuration

The system can be configured through `config.py`:

```python
# Agent configurations
AGENT_CONFIGS = {
    "coordinator": {
        "max_retries": 3,
        "timeout": 60,
        "confidence_threshold": 0.7
    },
    "research": {
        "max_sources": 20,
        "min_source_quality": 0.6,
        "search_depth": "comprehensive"
    },
    # ... other agent configs
}

# Quality control settings
QUALITY_METRICS = {
    "comprehensiveness_threshold": 0.8,
    "accuracy_threshold": 0.9,
    "originality_threshold": 0.7,
    "actionability_threshold": 0.8
}
```

## 🔧 API Usage

### Basic Research Query

```python
import asyncio
from system_orchestrator import SystemOrchestrator

async def run_research():
    orchestrator = SystemOrchestrator()
    await orchestrator.start_system()
    
    result = await orchestrator.execute_research_query(
        "What are the economic implications of vertical farming in urban environments?",
        options={
            "domain": "economics",
            "research_depth": "comprehensive"
        }
    )
    
    print(f"Research completed: {result['status']}")
    await orchestrator.stop_system()

asyncio.run(run_research())
```

### Custom Agent Configuration

```python
from agents.research_agent import ResearchAgent
from models import AgentType

# Create custom research agent
custom_config = {
    "max_sources": 30,
    "min_source_quality": 0.8,
    "search_depth": "deep"
}

research_agent = ResearchAgent()
research_agent.config.update(custom_config)
```

## 📊 Output Formats

### JSON Output Structure

```json
{
  "session_id": "uuid",
  "query": "Research question",
  "status": "completed",
  "results": {
    "coordinator": {
      "strategy": {...},
      "agent_sequence": [...],
      "confidence_score": 0.85
    },
    "research": {
      "sources": [...],
      "findings": [...],
      "confidence_score": 0.82
    },
    "analysis": {
      "patterns": [...],
      "correlations": [...],
      "insights": [...],
      "confidence_score": 0.78
    },
    "creative": {
      "concepts": [...],
      "innovation_metrics": {...},
      "confidence_score": 0.75
    },
    "validation": {
      "validation_report": {...},
      "quality_assessment": {...},
      "confidence_score": 0.88
    },
    "documentation": {
      "structured_content": {...},
      "format_options": {...},
      "confidence_score": 0.90
    }
  }
}
```

### Documentation Formats

The system generates documentation in multiple formats:

- **Markdown** - Human-readable format with headers, lists, and links
- **HTML** - Web-ready format with styling and interactive elements
- **JSON** - Machine-readable structured data
- **PDF** - Print-ready formatted documents

## 🎯 Use Cases

### Creative Research
- Design new languages, systems, or concepts
- Generate innovative solutions to complex problems
- Explore cross-domain applications and analogies

### Analytical Research
- Comprehensive market analysis
- Technology trend evaluation
- Economic impact assessment
- Scientific literature review

### Factual Research
- Fact-checking and verification
- Source credibility assessment
- Comprehensive topic coverage
- Multi-perspective analysis

## 🔍 Quality Assurance

### Validation Layers

1. **Factual Validation** - Source verification and cross-reference checking
2. **Logical Validation** - Consistency checking and contradiction detection
3. **Source Validation** - Credibility assessment and authority verification
4. **Bias Validation** - Perspective analysis and conflict of interest detection
5. **Completeness Validation** - Coverage analysis and gap identification

### Quality Metrics

- **Comprehensiveness** - Coverage of all relevant aspects
- **Accuracy** - Factual correctness and source reliability
- **Originality** - Novel insights and creative solutions
- **Actionability** - Practical implementation pathways
- **Efficiency** - Time to completion and resource utilization

## 🛠️ Development

### Project Structure

```
Autodisc/
├── agents/                 # Agent implementations
│   ├── __init__.py
│   ├── coordinator_agent.py
│   ├── research_agent.py
│   ├── analysis_agent.py
│   ├── creative_agent.py
│   ├── validation_agent.py
│   └── documentation_agent.py
├── config.py              # System configuration
├── models.py              # Data models and schemas
├── gemini_client.py       # Gemini API client
├── base_agent.py          # Base agent class
├── system_orchestrator.py # Main system orchestrator
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Adding New Agents

1. Create a new agent class inheriting from `BaseAgent`
2. Implement required abstract methods
3. Register the agent in the orchestrator
4. Update the agent sequence in coordinator strategies

### Extending Functionality

- **Custom Search APIs** - Extend research agent with new data sources
- **Specialized Analysis** - Add domain-specific analysis methods
- **Output Formats** - Implement new documentation formats
- **Quality Metrics** - Add custom validation criteria

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   ValueError: GEMINI_API_KEY is required in environment variables
   ```
   Solution: Set your Gemini API key in the `.env` file

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'google.generativeai'
   ```
   Solution: Install dependencies with `pip install -r requirements.txt`

3. **Timeout Errors**
   ```
   Research query timed out
   ```
   Solution: Increase timeout values in config.py or use simpler queries

### Debug Mode

Enable debug logging by modifying `config.py`:

```python
LOG_LEVEL = "DEBUG"
```

### Performance Optimization

- Adjust agent timeouts in configuration
- Reduce search depth for faster results
- Use parallel processing for large datasets
- Optimize API rate limiting

## 📈 Performance Metrics

The system tracks various performance metrics:

- **Agent Performance** - Success rates and processing times
- **Quality Scores** - Validation confidence and accuracy metrics
- **Resource Utilization** - Memory and CPU usage
- **Session Management** - Active sessions and completion rates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd Autodisc
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 .
black .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for AI capabilities
- The research community for inspiration
- Contributors and beta testers

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Autodisc** - Transforming research through autonomous multi-agent collaboration. 