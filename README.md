# AI Chat API Server

A FastAPI-based server providing a unified interface to multiple AI chat providers (GPT, Claude, and Gemini) with streaming responses.

## Features

- Multi-provider support (OpenAI/GPT, Anthropic/Claude, Google/Gemini)
- Server-Sent Events (SSE) streaming responses
- Automatic model fallback mechanism
- Comprehensive logging system
- Health monitoring endpoints
- Input validation
- Environment-based configuration

## Project Structure

```
.
├── main.py                 # FastAPI application entry point
├── aiproviders.py         # AI provider implementations
├── models.py              # Data models and validation
├── logging_config.py      # Logging configuration
├── .env                   # Environment variables (not in repo)
├── .env.example          # Environment template
├── docs/                  # Documentation directory
└── logs/                  # Log files directory
```

## Setup

0. Clone the repository, and navigate to the directory

1. Create and configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configurations
   ```

2. Install venv (recommended 3.11+)
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the server:
   ```bash
   python main.py
   ```

## Environment Configuration

Key configurations in `.env`:

```bash
# Server
PORT=3050

# API Keys
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key

# Models
OPENAI_MODEL_DEFAULT=gpt-4o
ANTHROPIC_MODEL_DEFAULT=claude-3-5-sonnet-latest
GEMINI_MODEL_DEFAULT=gemini-2.0-flash

# See .env.example for all configuration options
```

## API Endpoints

### Health Checks
```bash
# Overall health
GET /health

# Provider-specific health
GET /health/{provider}
```

### Chat
```bash
# Stream chat responses
POST /chat/{provider}

# Example request:
curl -X POST "http://localhost:3050/chat/gpt" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {"role": "user", "content": "Hello!"}
           ]
         }'
```

See [API Documentation](docs/api_docs.md) for detailed specifications.

## Model Selection Architecture

- Backend-controlled model selection
- Each provider has default and fallback models
- Model information included in response streams
- Automatic fallback on model failure
- Configuration via environment variables

## Logging System

Three-tier logging system:
- app.log: Information level
- error.log: Error level
- debug.log: Debug level with context

See [Logging Documentation](docs/logs_doc.md) for details.

## Development

### Adding New Provider
1. Add provider configuration to `.env.example`
2. Update `SUPPORTED_PROVIDERS` in environment
3. Add provider models to `PROVIDER_MODELS`
4. Implement provider-specific streaming in `aiproviders.py`
5. Update documentation

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. -v

# run with coverage and generate html report
pip install coverage

  
coverage run -m pytest


(Generate a Coverage Report:)
coverage report -m

coverage html
(Then open htmlcov/index.html in a browser.)
```

## Documentation

- [API Documentation](docs/api_docs.md)
- [Data Models](docs/data_models_docs.md)
- [File Structure](docs/file_structure_docs.md)
- [Logging System](docs/logs_doc.md)
- [Integration Examples](docs/api_integration_examples.md)
- [Test Documentation](docs/test_aiproviders_docs.md)

## Known Issues

1. Response Truncation:
   - In some cases, the beginning of the response may be truncated
   - Under investigation

2. Frontend/Backend Model Field:
   - Frontend may send 'model' field
   - Backend intentionally ignores model selection
   - Use model information from response streams

## License

MIT License

## Contributing

Feel free to contribute to the project by opening issues or submitting pull requests.

## Author

This project is developed by [Weiming](https://weiming.ai).

## Roadmap

- [ ] Add more providers
- [ ] Add more tests
- [ ] Add more documentation
- [ ] Add more examples

## Todo

- [ ] Known issue to fix: begging of the ai response truncation (not sure backend or frontend issue)
- [ ] todo 2 
- [ ] todo 3 
- [ ] todo 4 
- [ ] todo 5 
- [ ] todo 6 
- [ ] todo 7 
- [ ] todo 8 

## Last Updated

- 2025-02-21

Last updated: 

Last updated: 2025-02-22 09:51:24

Last updated: 2025-02-22 09:53:00

Last updated: 2025-02-22 10:03:38

Last updated: 2025-02-22 10:04:36

Last updated: 2025-02-22 10:06:21 UTC+0800

Last updated: 2025-02-22 10:41:46 UTC+0800 tiger lion elephant panda bear wolf fox deer snake monkey dragon phoenix cat dog horse rabbit eagle hawk owl bat shark whale dolphin seal penguin giraffe zebra rhino hippo leopard cheetah gorilla orangutan koala camel llama bison moose mars moon sun star galaxy nebula comet planet orbit asteroid blackhole supernova nova pulsar quark void cosmos ether spacetime gravity meteor plasma dust cluster

Last updated: 2025-02-22 10:47:20 UTC+0800 tiger lion elephant panda bear wolf fox deer snake monkey dragon phoenix cat dog horse rabbit eagle hawk owl bat shark whale dolphin seal penguin giraffe zebra rhino hippo leopard cheetah gorilla orangutan koala camel llama bison moose mars moon sun star galaxy nebula comet planet orbit asteroid blackhole supernova nova pulsar quark void cosmos ether spacetime gravity meteor plasma dust cluster
