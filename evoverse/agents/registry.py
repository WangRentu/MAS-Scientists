"""
Agent 注册表 - 管理所有活跃的 Agent
"""

from typing import Dict, Optional, List, Any
import logging

from .base_agent import BaseAgent


logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Agent 注册表
    
    管理所有活跃的 Agent 实例，提供查找和协调功能
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
    
    def register(self, agent: BaseAgent):
        """注册 Agent"""
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_id} already registered, replacing")
        
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_type} ({agent.agent_id})")
    
    def unregister(self, agent_id: str):
        """注销 Agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent {agent_id}")
        else:
            logger.warning(f"Agent {agent_id} not found for unregistration")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """获取 Agent"""
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """按类型获取 Agent"""
        return [agent for agent in self.agents.values() if agent.agent_type == agent_type]
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有 Agent"""
        return [
            {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "status": agent.status.value,
                "is_healthy": agent.is_healthy(),
                "created_at": agent.created_at.isoformat()
            }
            for agent in self.agents.values()
        ]
    
    def broadcast_message(self, message, exclude_agent: Optional[str] = None):
        """广播消息给所有 Agent"""
        for agent_id, agent in self.agents.items():
            if exclude_agent and agent_id == exclude_agent:
                continue
            agent.receive_message(message)
    
    def route_message(self, message):
        """路由消息到目标 Agent"""
        target_agent = self.get_agent(message.to_agent)
        if target_agent:
            target_agent.receive_message(message)
        else:
            logger.warning(f"No agent found for message routing: {message.to_agent}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取注册表统计"""
        total_agents = len(self.agents)
        agents_by_type = {}
        agents_by_status = {}
        
        for agent in self.agents.values():
            agents_by_type[agent.agent_type] = agents_by_type.get(agent.agent_type, 0) + 1
            agents_by_status[agent.status.value] = agents_by_status.get(agent.status.value, 0) + 1
        
        return {
            "total_agents": total_agents,
            "agents_by_type": agents_by_type,
            "agents_by_status": agents_by_status
        }


# 全局注册表实例
_agent_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """获取全局 Agent 注册表"""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry