import os
from typing import Literal, Optional

import openai
from langchain_core.language_models.chat_models import BaseChatModel

SourceType = Literal["OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom"]
ALLOWED_SOURCES: set[str] = set(SourceType.__args__)


def get_llm(
    model: str = "claude-4-sonnet-2025",
    temperature: float = 0.7,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str = 'custommodelfakeurl',
    api_key: str = "EMPTY",
) -> BaseChatModel:
    """
    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, Ollama, Gemini, Bedrock, and custom model serving.
    Args:
        model (str): The model name to use
        temperature (float): Temperature setting for generation
        stop_sequences (list): Sequences that will stop generation
        source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
                      If None, will attempt to auto-detect from model name
        base_url (str): The base URL for custom model serving (e.g., "http://localhost:8000/v1"), default is None
        api_key (str): The API key for the custom llm
    """
    # Auto-detect source from model name if not specified
    if source is None:
        import os
        env_source = os.getenv("LLM_SOURCE")
        if env_source in ALLOWED_SOURCES:
            source = env_source
        else:
            if model[:7] == "claude-":
                source = "Anthropic"
            elif model[:4] == "gpt-":
                source = "OpenAI"
            elif model.startswith("azure-"):
                source = "AzureOpenAI"
            elif model[:7] == "gemini-":
                source = "Gemini"
            elif "groq" in model.lower():
                source = "Groq"
            elif base_url is not None:
                source = "Custom"
            elif "/" in model or any(
                name in model.lower()
                for name in [
                    "llama",
                    "mistral",
                    "qwen",
                    "gemma",
                    "phi",
                    "dolphin",
                    "orca",
                    "vicuna",
                    "deepseek",
                    "gpt-oss",
                ]
            ):
                source = "Ollama"
            elif model.startswith(
                ("anthropic.claude-", "amazon.titan-", "meta.llama-", "mistral.", "cohere.", "ai21.", "us.")
            ):
                source = "Bedrock"
            else:
                raise ValueError("Unable to determine model source. Please specify 'source' parameter.")

    # Create appropriate model based on source
    if source == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(model=model, temperature=temperature, stop_sequences=stop_sequences)

    elif source == "AzureOpenAI":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Azure OpenAI models. Install with: pip install langchain-openai"
            )
        API_VERSION = "2024-12-01-preview"
        model = model.replace("azure-", "")
        return AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            azure_deployment=model,
            openai_api_version=API_VERSION,
            temperature=temperature,
        )

    elif source == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-anthropic package is required for Anthropic models. Install with: pip install langchain-anthropic"
            )
        import os
        final_url=os.environ.get("BASE_ENDPOINT") + '/sonnet-4/v1.0.0'
        final_key = os.environ.get("ANTHROPIC_KONGAPINYU_APIKEY")
        API_VERSION="bedrock-2023-05-31"
        return ChatAnthropic(
            model="claude-4-sonnet-2025",
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            base_url=final_url,
            api_key=final_key,
            extra_headers =  {"anthropic_version": API_VERSION}
        )

    elif source == "Gemini":
        # If you want to use ChatGoogleGenerativeAI, you need to pass the stop sequences upon invoking the model.
        # return ChatGoogleGenerativeAI(
        #     model=model,
        #     temperature=temperature,
        #     google_api_key=api_key,
        # )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Gemini models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            stop_sequences=stop_sequences,
        )

    elif source == "Groq":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Groq models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            stop_sequences=stop_sequences,
        )

    elif source == "Ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-ollama package is required for Ollama models. Install with: pip install langchain-ollama"
            )
        return ChatOllama(
            model=model,
            temperature=temperature,
        )

    elif source == "Bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-aws package is required for Bedrock models. Install with: pip install langchain-aws"
            )
        return ChatBedrock(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )

    elif source == "Custom":
        # Implement a proper ChatModel so llm.invoke([...]).content works
        from typing import Any, List
        import requests
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun
        import os
        class CustomClaudeChat(BaseChatModel):
            """Custom chat model that calls a REST endpoint.

            Returns AIMessage so callers can access `.content`.
            """

            model_route: str = "sonnet-4/v1.0.0"
            temperature: float = 0.7
            max_tokens: int = 8192*4
            base_endpoint: Optional[str] = None
            api_key: Optional[str] = None
            stop_sequences: Optional[List[str]] = None

            @property
            def _llm_type(self) -> str:  # pragma: no cover - required by BaseChatModel
                return "custom_claude_chat"

            @property
            def _identifying_params(self) -> dict:
                return {
                    "model_route": self.model_route,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "base_endpoint": self.base_endpoint,
                }

            def _generate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> ChatResult:
                # Resolve endpoint and key
                be = self.base_endpoint or os.environ.get("BASE_ENDPOINT")
                key = self.api_key or os.environ.get("ANTHROPIC_KONGAPINYU_APIKEY")
                if not be:
                    raise ValueError("CustomClaudeChat: base endpoint is not set")

                # Build endpoint as <base>/<model_route>
                base = (be or "").rstrip("/")
                route = "sonnet-4/v1.0.0"
                endpoint = f"{base}/{route}"
                headers = {
                    "Content-Type": "application/json",
                    "api-key": key or "",
                }

                # Add explicit system prompt handling
                system_prompt = ""
                if messages and messages[0].type == "system":
                    system_prompt = str(messages[0].content)
                    # Remove the system message from the list before converting
                    messages = messages[1:]
                # Convert LangChain messages to simple role/content pairs
                def role_of(msg: BaseMessage) -> str:
                    return "user" if msg.type == "human" else "assistant"

                converted_messages = [
                    {"role": role_of(m), "content": str(m.content)} for m in messages
                ]


                payload: dict[str, Any] = {
                    "messages": converted_messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    # Some backends expect this when proxying Anthropic
                    "anthropic_version": "bedrock-2023-05-31",
                }
                # Add system prompt if provided
                if system_prompt:
                    payload["system"] = system_prompt
                stop_list = self.stop_sequences
                if stop_list:
                    payload["stop_sequences"] = stop_list
                print(payload)
                try:
                    resp = requests.post(endpoint, headers=headers, json=payload, timeout=240)
                    resp.raise_for_status()
                    data = resp.json()

                    text: str = ""
                    # Handle Anthropic-style content list
                    if isinstance(data, dict) and "content" in data:
                        content = data.get("content")
                        if isinstance(content, list):
                            parts = [
                                part.get("text", "")
                                for part in content
                                if isinstance(part, dict) and part.get("type") == "text"
                            ]
                            text = "".join(parts)
                        elif isinstance(content, str):
                            text = content
                    # Handle OpenAI-style choices
                    if not text and isinstance(data, dict) and "choices" in data:
                        choices = data.get("choices") or []
                        if choices:
                            first = choices[0]
                            msg = (first.get("message") or {}).get("content") if isinstance(first, dict) else None
                            if isinstance(msg, str):
                                text = msg
                            elif isinstance(first, dict) and isinstance(first.get("text"), str):
                                text = first["text"]

                    if not text:
                        text = "No text content found"

                    ai = AIMessage(content=text)
                    generation = ChatGeneration(message=ai)
                    return ChatResult(generations=[generation])

                except Exception as e:  # noqa: BLE001
                    ai = AIMessage(content=f"Request failed: {e}")
                    generation = ChatGeneration(message=ai)
                    return ChatResult(generations=[generation])

        # Instantiate with provided configuration so default stop can be respected
        base_endpoint = os.environ.get("BASE_ENDPOINT")
        chat = CustomClaudeChat(
            model_route=model,
            temperature=temperature,
            max_tokens=8192,
            base_endpoint=os.environ.get("BASE_ENDPOINT"),
            api_key=os.environ.get("ANTHROPIC_KONGAPINYU_APIKEY"),
            stop_sequences=stop_sequences,
        )
        return chat
    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Groq', 'Bedrock', or 'Ollama'"
        )
