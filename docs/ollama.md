# Setting up Ollama for LLM Development

[Ollama](https://ollama.ai) allows you to run large language models locally, which is useful for development and testing HANI's LLM features without requiring API keys.

## Installing Ollama

### macOS

```bash
brew install ollama
```

Or download from [ollama.ai](https://ollama.ai/download)

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows

Download the installer from [ollama.ai](https://ollama.ai/download)

## Starting Ollama

Start the Ollama server:

```bash
ollama serve
```

This starts the server at `http://localhost:11434`.

## Pulling Models

Download a model to use with HANI:

```bash
# Recommended for development (small and fast)
ollama pull llama3.2

# Larger models for better quality
ollama pull llama3.1
ollama pull mistral
ollama pull codellama
```

## Configuring HANI for Ollama

Edit `~/negmas/hani/settings/llm_settings.json`:

```json
{
    "provider": "ollama",
    "model": "llama3.2",
    "base_url": "http://localhost:11434",
    "temperature": 0.7
}
```

### Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `provider` | LLM provider (`ollama`, `openai`, `anthropic`) | `openai` |
| `model` | Model name | `gpt-4o-mini` |
| `base_url` | API endpoint URL | Provider default |
| `temperature` | Response randomness (0.0-2.0) | `0.7` |
| `api_key` | API key (not needed for Ollama) | - |

## Testing the Configuration

1. Start Ollama:
   ```bash
   ollama serve
   ```

2. Start HANI:
   ```bash
   hani
   ```

3. Log in and start a negotiation

4. Use the **Response Generator** tool to test LLM integration:
   - Enter an instruction like "Propose a fair offer"
   - Click **Generate Response**
   - The generated text and outcome should appear

## Recommended Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `llama3.2` | 3B | Fast | Good | Development, testing |
| `llama3.1` | 8B | Medium | Better | General use |
| `mistral` | 7B | Medium | Good | Balanced performance |
| `llama3.1:70b` | 70B | Slow | Excellent | Production quality |

## Troubleshooting

### Connection Refused

If you see "Connection refused" errors:

1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

2. Check the server is accessible:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Model Not Found

If the model isn't found:

1. List available models:
   ```bash
   ollama list
   ```

2. Pull the model if needed:
   ```bash
   ollama pull llama3.2
   ```

### Slow Responses

For faster responses:

1. Use a smaller model (`llama3.2` instead of `llama3.1:70b`)
2. Reduce the context window in LLM settings
3. Ensure no other heavy processes are running

## Using OpenAI or Anthropic Instead

If you prefer cloud-based models:

### OpenAI

```json
{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "sk-..."
}
```

Or set the environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### Anthropic

```json
{
    "provider": "anthropic",
    "model": "claude-3-haiku-20240307",
    "api_key": "sk-ant-..."
}
```

Or set the environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Admin LLM Settings

Administrators can configure LLM settings directly in the HANI interface:

1. Log in as admin
2. Open the sidebar
3. Navigate to **LLM Settings**
4. Configure provider, model, and prompts

Changes are saved to `llm_settings.json` and take effect immediately.
