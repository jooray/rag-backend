from typing import Optional, Tuple, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from ..models.config import (
    PipelineConfig,
    PromptConfig,
    GatePromptConfig,
    RewritePromptConfig,
    FixPromptConfig,
    ModelConfig,
)


class PipelineService:
    # Service is stateless across invocations (no shared mutable per-request state), safe for thread pool usage
    def __init__(self, config: PipelineConfig, models: Dict[str, ModelConfig], api_key: str, api_base: str):
        self.config = config
        self.models = models
        self.api_key = api_key
        self.api_base = api_base

    def _get_model_config(self, model_id: str) -> ModelConfig:
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found in models configuration")
        return self.models[model_id]

    def _get_llm(self, model_id: str) -> ChatOpenAI:
        model_config = self._get_model_config(model_id)
        return ChatOpenAI(
            model=model_config.name,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base,
        )

    def _convert_messages_to_langchain(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """Convert OpenAI format messages to LangChain format"""
        langchain_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        return langchain_messages

    def _run_prompt_with_history(self, prompt_config: PromptConfig, messages: List[Dict[str, str]], **kwargs) -> str:
        """Run prompt with conversation history for main inference"""
        llm = self._get_llm(prompt_config.model)

        # Convert conversation history to LangChain format
        conversation_messages = self._convert_messages_to_langchain(messages[:-1])  # All but last message

        # Add system prompt
        system_message = SystemMessage(content=prompt_config.system_prompt)
        all_messages = [system_message] + conversation_messages

        # Format the last user message with context
        last_user_message = messages[-1]
        user_content = prompt_config.user_prompt_template.format(
            question=last_user_message.get("content", ""),
            **kwargs
        )
        all_messages.append(HumanMessage(content=user_content))

        response = llm.invoke(all_messages)
        return response.content

    def _run_prompt(self, prompt_config: PromptConfig, **kwargs) -> str:
        llm = self._get_llm(prompt_config.model)

        system_message = SystemMessage(content=prompt_config.system_prompt)
        user_content = prompt_config.user_prompt_template.format(**kwargs)
        user_message = HumanMessage(content=user_content)

        response = llm.invoke([system_message, user_message])
        return response.content

    def _run_gate_prompt(
        self, gate_config: GatePromptConfig, response: str
    ) -> Tuple[bool, Optional[str]]:
        llm = self._get_llm(gate_config.model)

        system_message = SystemMessage(content=gate_config.system_prompt)
        user_content = gate_config.user_prompt_template.format(response=response)
        user_message = HumanMessage(content=user_content)

        result = llm.invoke([system_message, user_message]).content.strip()

        if result.upper().startswith("PASS"):
            return True, None
        elif result.upper().startswith("REJECT"):
            reject_reason = (
                result[6:].strip() if len(result) > 6 else "No reason provided"
            )
            return False, reject_reason
        else:
            return True, None

    def _run_fix_prompt(
        self, fix_config: FixPromptConfig, response: str, reject_reason: str
    ) -> str:
        llm = self._get_llm(fix_config.model)

        system_message = SystemMessage(content=fix_config.system_prompt)
        user_content = fix_config.user_prompt_template.format(
            response=response, reject_reason=reject_reason
        )
        user_message = HumanMessage(content=user_content)

        return llm.invoke([system_message, user_message]).content

    def _run_rewrite_prompt(
        self, rewrite_config: RewritePromptConfig, response: str
    ) -> str:
        llm = self._get_llm(rewrite_config.model)

        system_message = SystemMessage(content=rewrite_config.system_prompt)
        user_content = rewrite_config.user_prompt_template.format(response=response)
        user_message = HumanMessage(content=user_content)

        return llm.invoke([system_message, user_message]).content

    def run_pipeline(self, messages: List[Dict[str, str]], context: str) -> str:
        # Use conversation history for main prompt
        response = self._run_prompt_with_history(
            self.config.main_prompt, messages, context=context
        )

        for attempt in range(self.config.max_retries):
            all_gates_passed = True

            for gate_config in self.config.gate_prompts:
                passed, reject_reason = self._run_gate_prompt(gate_config, response)

                if not passed:
                    all_gates_passed = False
                    if gate_config.fix_prompt:
                        response = self._run_fix_prompt(
                            gate_config.fix_prompt, response, reject_reason
                        )
                    break

            if all_gates_passed:
                break

        for rewrite_config in self.config.rewrite_prompts:
            response = self._run_rewrite_prompt(rewrite_config, response)

        return response
