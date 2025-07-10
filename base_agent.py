"""
Base agent class for the Multi-Agent Autonomous Research System
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from models import AgentType, AgentMessage, AgentOutput, MessagePriority
from gemini_client import GeminiClient
from config import Config

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_type: AgentType, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent
        
        Args:
            agent_type: Type of this agent
            config: Agent-specific configuration
        """
        self.agent_type = agent_type
        self.config = config or Config.get_agent_config(agent_type.value)
        self.gemini_client = GeminiClient()
        self.message_queue: List[AgentMessage] = []
        self.processing_history: List[Dict[str, Any]] = []
        self.is_active = False
        
        logger.info(f"Initialized {agent_type.value} agent")
    
    async def start(self):
        """Start the agent's processing loop"""
        self.is_active = True
        logger.info(f"{self.agent_type.value} agent started")
    
    async def stop(self):
        """Stop the agent's processing loop"""
        self.is_active = False
        logger.info(f"{self.agent_type.value} agent stopped")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentOutput]:
        """
        Process an incoming message and generate a response
        
        Args:
            message: Incoming message to process
            
        Returns:
            Agent output or None if no response needed
        """
        start_time = time.time()
        
        try:
            # Add message to queue
            self.message_queue.append(message)
            
            # Process the message based on type
            if message.message_type == "task":
                output = await self._process_task(message)
            elif message.message_type == "query":
                output = await self._process_query(message)
            elif message.message_type == "feedback":
                output = await self._process_feedback(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                output = None
            
            # Record processing history
            processing_time = time.time() - start_time
            self.processing_history.append({
                "message_id": message.id,
                "message_type": message.message_type,
                "processing_time": processing_time,
                "timestamp": datetime.now(),
                "success": output is not None
            })
            
            return output
            
        except Exception as e:
            logger.error(f"Error processing message in {self.agent_type.value}: {str(e)}")
            processing_time = time.time() - start_time
            
            # Record error in history
            self.processing_history.append({
                "message_id": message.id,
                "message_type": message.message_type,
                "processing_time": processing_time,
                "timestamp": datetime.now(),
                "success": False,
                "error": str(e)
            })
            
            return None
    
    async def _process_task(self, message: AgentMessage) -> Optional[AgentOutput]:
        """Process a task message - to be implemented by subclasses"""
        return await self.execute_task(message.content)
    
    async def _process_query(self, message: AgentMessage) -> Optional[AgentOutput]:
        """Process a query message - to be implemented by subclasses"""
        return await self.execute_query(message.content)
    
    async def _process_feedback(self, message: AgentMessage) -> Optional[AgentOutput]:
        """Process a feedback message - to be implemented by subclasses"""
        return await self.process_feedback(message.content)
    
    @abstractmethod
    async def execute_task(self, task_data: Dict[str, Any]) -> Optional[AgentOutput]:
        """
        Execute a specific task - to be implemented by subclasses
        
        Args:
            task_data: Task-specific data
            
        Returns:
            Agent output or None
        """
        pass
    
    @abstractmethod
    async def execute_query(self, query_data: Dict[str, Any]) -> Optional[AgentOutput]:
        """
        Execute a query - to be implemented by subclasses
        
        Args:
            query_data: Query-specific data
            
        Returns:
            Agent output or None
        """
        pass
    
    async def process_feedback(self, feedback_data: Dict[str, Any]) -> Optional[AgentOutput]:
        """
        Process feedback - default implementation
        
        Args:
            feedback_data: Feedback data
            
        Returns:
            Agent output or None
        """
        logger.info(f"{self.agent_type.value} received feedback: {feedback_data}")
        return None
    
    def create_message(
        self, 
        recipient: AgentType, 
        message_type: str, 
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        confidence_score: Optional[float] = None
    ) -> AgentMessage:
        """
        Create a new message to send to another agent
        
        Args:
            recipient: Target agent
            message_type: Type of message
            content: Message content
            priority: Message priority
            confidence_score: Confidence score for the message
            
        Returns:
            New agent message
        """
        return AgentMessage(
            sender=self.agent_type,
            recipient=recipient,
            message_type=message_type,
            content=content,
            priority=priority,
            confidence_score=confidence_score
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_type": self.agent_type.value,
            "is_active": self.is_active,
            "queue_size": len(self.message_queue),
            "processing_history_count": len(self.processing_history),
            "config": self.config
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent"""
        if not self.processing_history:
            return {
                "total_messages": 0,
                "success_rate": 0.0,
                "average_processing_time": 0.0,
                "error_rate": 0.0
            }
        
        total_messages = len(self.processing_history)
        successful_messages = sum(1 for record in self.processing_history if record.get("success", False))
        error_messages = total_messages - successful_messages
        
        processing_times = [record.get("processing_time", 0) for record in self.processing_history]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "total_messages": total_messages,
            "success_rate": successful_messages / total_messages if total_messages > 0 else 0.0,
            "average_processing_time": avg_processing_time,
            "error_rate": error_messages / total_messages if total_messages > 0 else 0.0
        }
    
    async def generate_response_with_gemini(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the Gemini API
        
        Args:
            prompt: The prompt to send
            context: Additional context
            system_prompt: System-level instructions
            
        Returns:
            Response from Gemini API
        """
        return await self.gemini_client.generate_response(
            prompt=prompt,
            context=context,
            system_prompt=system_prompt
        )
    
    async def generate_structured_response_with_gemini(
        self, 
        prompt: str, 
        output_format: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured response using the Gemini API
        
        Args:
            prompt: The prompt to send
            output_format: Desired output structure
            context: Additional context
            
        Returns:
            Structured response from Gemini API
        """
        return await self.gemini_client.generate_structured_response(
            prompt=prompt,
            output_format=output_format,
            context=context
        )
    
    def calculate_confidence_score(self, factors: Dict[str, float]) -> float:
        """
        Calculate a confidence score based on multiple factors
        
        Args:
            factors: Dictionary of factors and their weights
            
        Returns:
            Calculated confidence score
        """
        if not factors:
            return 0.5  # Default confidence
        
        total_weight = sum(factors.values())
        if total_weight == 0:
            return 0.5
        
        weighted_sum = sum(score * weight for score, weight in factors.items())
        confidence = weighted_sum / total_weight
        
        return max(0.0, min(1.0, confidence))
    
    def log_activity(self, activity: str, details: Optional[Dict[str, Any]] = None):
        """Log agent activity"""
        log_message = f"{self.agent_type.value}: {activity}"
        if details:
            log_message += f" - {details}"
        logger.info(log_message) 