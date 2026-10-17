# Model Fallback Chain Implementation

## Overview
Implemented automatic model fallback chain that switches to alternative models when the primary model exceeds its rate limit.

## Implementation Details

### Location
`backend/agent_graph.py`

### Configuration
```python
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
FALLBACK_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
]
```

### How It Works

1. **Retry Logic**: When a model call fails, the system retries up to 4 times with exponential backoff (2s, 4s, 8s, 16s)

2. **Fallback Trigger**: If all retries fail due to rate limit (`TooManyRequestsResponseError`), the system automatically switches to the next model in `FALLBACK_MODELS`

3. **Model Switching**: 
   - Creates a new `ChatOpenRouter` instance for the fallback model
   - Binds tools to the new model
   - Updates the global `model` and `model_with_tools` variables
   - Retries the request from scratch with the new model

4. **User Notification**: When a model switch occurs, an `AIMessage` is added to the conversation:
   ```
   Note: The model 'nvidia/nemotron-3-super-120b-a12b:free' exceeded its rate limit. 
   Switched to 'meta-llama/llama-3.2-3b-instruct:free' to continue.
   ```

### Code Changes

#### New Functions
- `_create_model(model_name)`: Creates a ChatOpenRouter instance
- `_switch_to_next_model()`: Switches to the next fallback model in the chain

#### Modified Functions
- `invoke_model()`: Now returns `(response, switched_model_name_or_None)` tuple
  - Tries current model first with retries
  - On rate limit exhaustion, switches to next fallback model
  - Continues trying fallbacks until one succeeds or all are exhausted
- `agent_node()`: Handles the new return tuple and injects notification message
- `extract_memory_node()`, `refine_node()`, `fallback_node()`: Updated to handle tuple return

### State Tracking
```python
_current_model_index = 0  # Tracks which model is currently active
_all_models = [MODEL_NAME] + FALLBACK_MODELS  # Full chain
```

## Current Limitations

### Free Model Availability
OpenRouter's free models have significant limitations:
1. **Rate Limits**: Very restrictive (50 requests/day, 20 RPM account-wide)
2. **Tool Support**: Many free models don't support function calling
3. **Deprecation**: Free models are frequently removed or changed
4. **Reliability**: Free tier is not suitable for production use

### Observed Issues
- Primary model (`nvidia/nemotron-3-super-120b-a12b:free`): Rate limited after extensive testing
- Fallback model (`nvidia/nemotron-3.5-content-safety:free`): Doesn't support tool calling
- Fallback model (`google/gemma-3-1b-it:free`): Model slug doesn't exist
- Fallback model (`meta-llama/llama-3.2-3b-instruct:free`): No longer available for free

## Recommendations

### For Development/Testing
1. **Add Credits**: Add $10+ to OpenRouter account to unlock 1000 free requests/day
2. **Use Paid Models**: Switch to paid versions (remove `:free` suffix) for reliability
3. **Alternative Providers**: Consider using OpenAI, Anthropic, or Google directly

### For Production
1. **Multiple Providers**: Implement fallback across different providers (OpenAI, Anthropic, etc.)
2. **Paid Tier**: Use paid models with proper rate limits
3. **Monitoring**: Add metrics to track model switches and failures
4. **Caching**: Implement response caching to reduce API calls

## Testing the Fallback

To test the fallback mechanism:
1. Exhaust the primary model's rate limit
2. Send a request to `/chat/hitl`
3. Check server logs for:
   ```
   [retry] TooManyRequestsResponseError on attempt 1/4; waiting 2.6s before retry
   [retry] TooManyRequestsResponseError on attempt 2/4; waiting 5.0s before retry
   [retry] TooManyRequestsResponseError on attempt 3/4; waiting 8.8s before retry
   [fallback] Switching from model-a to model-b
   [fallback] Rate limit exceeded on model-a, shifted to model-b
   ```
4. Check the response for the notification message about model switching

## Future Improvements

1. **Provider Abstraction**: Support multiple LLM providers (OpenAI, Anthropic, etc.)
2. **Health Checks**: Periodically check model availability before switching
3. **Smart Selection**: Choose fallback based on model capabilities (tool support, context length)
4. **Cost Tracking**: Track costs across different models
5. **Admin API**: Allow dynamic configuration of fallback chain via API
