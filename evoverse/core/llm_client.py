# evoverse/core/llm_client.py
from typing import List, Dict, Any, Optional
from openai import OpenAI
from datetime import datetime
import logging
import json
import re
from evoverse.config import get_config
logger = logging.getLogger(__name__)

# Type alias for OpenAI-style messages
Message = Dict[str, str]


def create_message(role: str, content: str) -> Message:
    """Create a message dict."""
    return {"role": role, "content": content}


def _summarize_messages(messages: List[Message], max_chars: int = 200) -> str:
    """Return a compact representation of the messages for logging."""
    summary_parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        snippet = content[:max_chars]
        if len(content) > max_chars:
            snippet += "..."
        summary_parts.append(f"{role}: {snippet}")
    return " | ".join(summary_parts)

class ConversationMemory:
    """对话记忆管理器"""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
    
    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        self.messages.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
        
        # 限制历史长度
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        self.last_accessed = datetime.now()
    
    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """获取消息历史（不包含时间戳）"""
        msgs = []
        for msg in self.messages:
            if not include_system and msg["role"] == "system":
                continue
            msgs.append({"role": msg["role"], "content": msg["content"]})
        return msgs
    
    def clear_history(self):
        """清空对话历史"""
        self.messages = []
        self.last_accessed = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "total_messages": len(self.messages),
            "max_history": self.max_history,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat()
        }


class LLMClient:
    """
    增强的 LLM 客户端，支持多轮对话记忆
    """
    
    def __init__(self, max_history: int = 50):
        cfg = get_config().llm
        self.model = cfg.model
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
        )
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.timeout = cfg.request_timeout
        
        # 对话记忆
        self.conversation_memory = ConversationMemory(max_history)
        
        # 统计信息
        self.total_requests = 0
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        发送消息到 LLM（兼容原有接口）
        """
        request_id = f"llm-{self.total_requests + 1}"
        logger.info(
            "LLM chat start | request_id=%s model=%s messages=%d",
            request_id,
            self.model,
            len(messages),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("LLM chat payload | request_id=%s %s", request_id, _summarize_messages(messages))

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        except Exception:
            logger.warning("LLM chat failed | request_id=%s", request_id, exc_info=True)
            raise

        reply = resp.choices[0].message.content
        reply = re.sub(r".*?</think>", "", reply, flags=re.DOTALL).strip()
        self.total_requests += 1

        logger.info(
            "LLM chat success | request_id=%s tokens_est=%s total_requests=%d",
            request_id,
            getattr(resp, "usage", None),
            self.total_requests,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("LLM chat reply | request_id=%s %s", request_id, reply)

        return reply

    def generate_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        生成结构化输出 (JSON)

        Args:
            prompt: 用户提示
            output_schema: 输出结构模式
            system: 系统提示
            max_tokens: 最大token数

        Returns:
            dict: 解析后的JSON响应
        """
        # 添加JSON指令到系统提示
        json_instruction = "\n\n你必须返回符合以下JSON格式的有效响应：\n" + json.dumps(output_schema, indent=2, ensure_ascii=False) + "\n\n必须用 ```json ... ``` 代码块包裹着JSON结果。"
        json_system = (system or "") + json_instruction

        # 构建消息
        messages = []
        if json_system:
            messages.append({"role": "system", "content": json_system})
        messages.append({"role": "user", "content": prompt})

        logger.info("LLM structured generation start")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("LLM structured schema: %s", json.dumps(output_schema, ensure_ascii=False))

        # 生成响应
        response_text = self.chat(messages)
        origin_response = response_text

        # 解析JSON
        try:
            # 尝试从markdown代码块中提取JSON（如果存在）
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                if json_end == -1:
                    json_end = len(response_text)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                if json_end == -1:
                    json_end = len(response_text)
                response_text = response_text[json_start:json_end].strip()

            # 解析JSON
            return json.loads(response_text)

        except json.JSONDecodeError as e:
            # 如果解析失败，返回基本结构
            logger.error(f"JSON解析失败: {e}, \n\n响应内容: {origin_response}")
            # 返回schema中的默认值
            return self._get_default_from_schema(output_schema)
    
    def chat_with_memory(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        带记忆的对话接口
        自动管理对话历史，支持连续对话
        """
        # 添加用户消息到记忆
        self.conversation_memory.add_message("user", user_message)
        
        # 构建完整消息历史
        messages = []
        
        # 添加系统提示（如果提供）
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加历史消息
        messages.extend(self.conversation_memory.get_messages(include_system=False))
        
        # 发送到 LLM
        reply = self.chat(messages)
        
        # 将回复添加到记忆
        self.conversation_memory.add_message("assistant", reply)
        
        return reply
    
    def set_system_prompt(self, prompt: str):
        """设置系统提示（会覆盖之前的系统消息）"""
        # 移除旧的系统消息
        self.conversation_memory.messages = [
            msg for msg in self.conversation_memory.messages 
            if msg["role"] != "system"
        ]
        # 添加新的系统消息
        self.conversation_memory.add_message("system", prompt)
    
    def clear_memory(self):
        """清空对话记忆"""
        self.conversation_memory.clear_history()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            "conversation": self.conversation_memory.get_stats(),
            "total_requests": self.total_requests
        }

    def _get_default_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """从JSON schema生成默认值"""
        if schema.get("type") == "object":
            result = {}
            properties = schema.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_schema.get("type") == "string":
                    result[prop_name] = ""
                elif prop_schema.get("type") == "number":
                    result[prop_name] = 0.0
                elif prop_schema.get("type") == "integer":
                    result[prop_name] = 0
                elif prop_schema.get("type") == "boolean":
                    result[prop_name] = False
                elif prop_schema.get("type") == "array":
                    result[prop_name] = []
                else:
                    result[prop_name] = None
            return result
        elif schema.get("type") == "array":
            return []
        else:
            return {}
