"""
Coordinator Agent for the Multi-Agent Autonomous Research System
"""
import asyncio
from typing import Dict, List, Any, Optional
import logging

from base_agent import BaseAgent
from models import (
    AgentType, AgentMessage, CoordinatorOutput, ResearchContext, 
    ResearchSession, MessagePriority
)
from config import Config

logger = logging.getLogger(__name__)

class CoordinatorAgent(BaseAgent):
    """Master controller that orchestrates the entire research workflow"""
    
    def __init__(self):
        """Initialize the coordinator agent"""
        super().__init__(AgentType.COORDINATOR)
        self.active_sessions: Dict[str, ResearchSession] = {}
        self.agent_registry: Dict[AgentType, Any] = {}
        self.research_strategies = self._load_research_strategies()
        
    def _load_research_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined research strategies"""
        return {
            "creative": {
                "agent_sequence": [
                    AgentType.RESEARCH,
                    AgentType.CREATIVE,
                    AgentType.ANALYSIS,
                    AgentType.VALIDATION,
                    AgentType.DOCUMENTATION
                ],
                "research_depth": "comprehensive",
                "creativity_focus": True,
                "validation_layers": ["logical", "feasibility", "innovation"]
            },
            "analytical": {
                "agent_sequence": [
                    AgentType.RESEARCH,
                    AgentType.ANALYSIS,
                    AgentType.VALIDATION,
                    AgentType.DOCUMENTATION
                ],
                "research_depth": "comprehensive",
                "analysis_focus": True,
                "validation_layers": ["factual", "logical", "source"]
            },
            "factual": {
                "agent_sequence": [
                    AgentType.RESEARCH,
                    AgentType.VALIDATION,
                    AgentType.DOCUMENTATION
                ],
                "research_depth": "standard",
                "validation_focus": True,
                "validation_layers": ["factual", "source"]
            }
        }
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Optional[CoordinatorOutput]:
        """Execute a research task"""
        query = task_data.get("query", "")
        session_id = task_data.get("session_id")
        
        if not query:
            logger.error("No query provided for research task")
            return None
        
        # Create or get research session
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
        else:
            session = await self._create_research_session(query)
            session_id = session.id
            self.active_sessions[session_id] = session
        
        # Analyze query and determine strategy
        strategy = await self._analyze_query_and_strategy(query)
        
        # Execute research workflow
        result = await self._execute_research_workflow(session, strategy)
        
        return CoordinatorOutput(
            agent_type=AgentType.COORDINATOR,
            content=result,
            confidence_score=result.get("confidence_score", 0.8),
            strategy=strategy,
            agent_sequence=strategy.get("agent_sequence", []),
            resource_allocation=result.get("resource_allocation", {}),
            quality_gates=result.get("quality_gates", [])
        )
    
    async def execute_query(self, query_data: Dict[str, Any]) -> Optional[CoordinatorOutput]:
        """Execute a query about system status or session information"""
        query_type = query_data.get("type", "status")
        
        if query_type == "status":
            return await self._get_system_status()
        elif query_type == "session":
            session_id = query_data.get("session_id")
            return await self._get_session_status(session_id)
        elif query_type == "performance":
            return await self._get_performance_metrics()
        else:
            logger.warning(f"Unknown query type: {query_type}")
            return None
    
    async def _create_research_session(self, query: str) -> ResearchSession:
        """Create a new research session"""
        # Analyze query complexity
        complexity_analysis = await self._analyze_query_complexity(query)
        
        # Determine domain
        domain = await self._identify_domain(query)
        
        # Create research context
        context = ResearchContext(
            query=query,
            domain=domain,
            complexity_score=complexity_analysis.get("complexity_score", 0.5),
            research_depth="comprehensive",
            output_requirements=complexity_analysis.get("output_requirements", {}),
            constraints=complexity_analysis.get("constraints", []),
            success_criteria=complexity_analysis.get("success_criteria", [])
        )
        
        # Create session
        session = ResearchSession(context=context)
        
        logger.info(f"Created research session {session.id} for query: {query}")
        return session
    
    async def _analyze_query_and_strategy(self, query: str) -> Dict[str, Any]:
        """Analyze query and determine optimal research strategy"""
        prompt = f"""
        Analyze the following research query and determine the optimal research strategy:
        
        Query: {query}
        
        Consider:
        1. Query complexity and domain
        2. Whether it's creative, analytical, or factual
        3. Required research depth
        4. Optimal agent sequence
        5. Quality gates and success criteria
        
        Provide a structured strategy recommendation.
        """
        
        output_format = {
            "strategy_type": "string (creative/analytical/factual)",
            "complexity_score": "float (0.0-1.0)",
            "domain": "string",
            "research_depth": "string (standard/comprehensive/deep)",
            "agent_sequence": "list of agent types",
            "quality_gates": "list of quality checkpoints",
            "estimated_duration": "integer (minutes)",
            "resource_requirements": "object"
        }
        
        response = await self.generate_structured_response_with_gemini(
            prompt=prompt,
            output_format=output_format
        )
        
        if response.get("success") and response.get("structured_data"):
            strategy_data = response["structured_data"]
            
            # Map to predefined strategy if possible
            strategy_type = strategy_data.get("strategy_type", "analytical")
            base_strategy = self.research_strategies.get(strategy_type, self.research_strategies["analytical"])
            
            # Merge with base strategy
            strategy = {**base_strategy, **strategy_data}
            
            return strategy
        else:
            # Fallback to analytical strategy
            return self.research_strategies["analytical"]
    
    async def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze the complexity of a research query"""
        prompt = f"""
        Analyze the complexity of this research query:
        
        Query: {query}
        
        Evaluate:
        1. Complexity score (0.0-1.0)
        2. Required output format
        3. Constraints and limitations
        4. Success criteria
        5. Potential challenges
        """
        
        output_format = {
            "complexity_score": "float (0.0-1.0)",
            "output_requirements": "object",
            "constraints": "list of strings",
            "success_criteria": "list of strings",
            "challenges": "list of strings"
        }
        
        response = await self.generate_structured_response_with_gemini(
            prompt=prompt,
            output_format=output_format
        )
        
        if response.get("success") and response.get("structured_data"):
            return response["structured_data"]
        else:
            return {
                "complexity_score": 0.5,
                "output_requirements": {},
                "constraints": [],
                "success_criteria": ["comprehensive coverage", "accurate information"],
                "challenges": []
            }
    
    async def _identify_domain(self, query: str) -> str:
        """Identify the domain of the research query"""
        prompt = f"""
        Identify the primary domain of this research query:
        
        Query: {query}
        
        Choose from: technology, science, business, arts, humanities, social_sciences, 
        engineering, medicine, education, environment, politics, economics, or other
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            content = response["content"].lower()
            domains = [
                "technology", "science", "business", "arts", "humanities", 
                "social_sciences", "engineering", "medicine", "education", 
                "environment", "politics", "economics"
            ]
            
            for domain in domains:
                if domain in content:
                    return domain
            
            return "other"
        
        return "general"
    
    async def _execute_research_workflow(
        self, 
        session: ResearchSession, 
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the complete research workflow"""
        self.log_activity("Starting research workflow", {"session_id": session.id})
        
        agent_sequence = strategy.get("agent_sequence", [])
        quality_gates = strategy.get("quality_gates", [])
        
        # Initialize workflow state
        workflow_state = {
            "session_id": session.id,
            "current_phase": 0,
            "completed_phases": [],
            "results": {},
            "quality_checks": [],
            "confidence_score": 0.0
        }
        
        # Execute each phase
        for i, agent_type in enumerate(agent_sequence):
            self.log_activity(f"Executing phase {i+1}", {"agent": agent_type.value})
            
            # Execute agent task
            agent_result = await self._execute_agent_task(agent_type, session, workflow_state)
            
            if agent_result:
                workflow_state["results"][agent_type.value] = agent_result
                workflow_state["completed_phases"].append(agent_type.value)
                
                # Update confidence score
                workflow_state["confidence_score"] = self._update_confidence_score(
                    workflow_state["confidence_score"], 
                    agent_result.get("confidence_score", 0.0)
                )
            
            # Check quality gates
            if quality_gates and i < len(quality_gates):
                quality_check = await self._check_quality_gate(
                    quality_gates[i], workflow_state
                )
                workflow_state["quality_checks"].append(quality_check)
                
                if not quality_check.get("passed", True):
                    self.log_activity("Quality gate failed", {"gate": quality_gates[i]})
                    break
        
        # Finalize session
        session.status = "completed"
        session.quality_metrics = workflow_state.get("quality_checks", [])
        
        return {
            "workflow_state": workflow_state,
            "session": session,
            "confidence_score": workflow_state["confidence_score"],
            "resource_allocation": self._calculate_resource_allocation(workflow_state),
            "quality_gates": workflow_state["quality_checks"]
        }
    
    async def _execute_agent_task(
        self, 
        agent_type: AgentType, 
        session: ResearchSession, 
        workflow_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute a task with a specific agent"""
        # Create task message
        task_data = {
            "session_id": session.id,
            "query": session.context.query,
            "context": session.context.dict(),
            "workflow_state": workflow_state,
            "previous_results": workflow_state.get("results", {})
        }
        
        message = self.create_message(
            recipient=agent_type,
            message_type="task",
            content=task_data,
            priority=MessagePriority.HIGH
        )
        
        # Send message to agent (in a real implementation, this would go through a message bus)
        # For now, we'll simulate the response
        self.log_activity(f"Sending task to {agent_type.value}", {"task_data": task_data})
        
        # Simulate agent processing
        await asyncio.sleep(1)  # Simulate processing time
        
        # Return simulated result
        return {
            "agent_type": agent_type.value,
            "status": "completed",
            "confidence_score": 0.8,
            "result": f"Simulated result from {agent_type.value}",
            "processing_time": 1.0
        }
    
    async def _check_quality_gate(
        self, 
        gate: str, 
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if a quality gate passes"""
        prompt = f"""
        Evaluate if the research workflow passes this quality gate:
        
        Gate: {gate}
        Current state: {workflow_state}
        
        Determine if the quality criteria are met.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            content = response["content"].lower()
            passed = "pass" in content or "success" in content or "meet" in content
            
            return {
                "gate": gate,
                "passed": passed,
                "evaluation": response["content"],
                "confidence_score": response.get("confidence_score", 0.7)
            }
        else:
            return {
                "gate": gate,
                "passed": True,  # Default to pass if evaluation fails
                "evaluation": "Evaluation failed",
                "confidence_score": 0.5
            }
    
    def _update_confidence_score(self, current_score: float, new_score: float) -> float:
        """Update the overall confidence score"""
        # Weighted average with more weight on current score
        return (current_score * 0.7) + (new_score * 0.3)
    
    def _calculate_resource_allocation(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource allocation based on workflow state"""
        completed_phases = len(workflow_state.get("completed_phases", []))
        total_phases = len(workflow_state.get("results", {}))
        
        return {
            "completed_phases": completed_phases,
            "total_phases": total_phases,
            "completion_percentage": (completed_phases / total_phases * 100) if total_phases > 0 else 0,
            "estimated_remaining_time": max(0, total_phases - completed_phases) * 2  # minutes
        }
    
    async def _get_system_status(self) -> Optional[CoordinatorOutput]:
        """Get overall system status"""
        status = {
            "active_sessions": len(self.active_sessions),
            "agent_registry": len(self.agent_registry),
            "system_health": "operational",
            "performance_metrics": self.get_performance_metrics()
        }
        
        return CoordinatorOutput(
            agent_type=AgentType.COORDINATOR,
            content=status,
            confidence_score=1.0,
            strategy={},
            agent_sequence=[],
            resource_allocation=status,
            quality_gates=[]
        )
    
    async def _get_session_status(self, session_id: str) -> Optional[CoordinatorOutput]:
        """Get status of a specific session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return CoordinatorOutput(
            agent_type=AgentType.COORDINATOR,
            content={"session": session.dict()},
            confidence_score=1.0,
            strategy={},
            agent_sequence=[],
            resource_allocation={},
            quality_gates=[]
        )
    
    async def _get_performance_metrics(self) -> Optional[CoordinatorOutput]:
        """Get performance metrics"""
        metrics = {
            "coordinator_metrics": self.get_performance_metrics(),
            "active_sessions_count": len(self.active_sessions),
            "average_session_duration": 0,  # Would calculate from session data
            "success_rate": 0.9  # Would calculate from historical data
        }
        
        return CoordinatorOutput(
            agent_type=AgentType.COORDINATOR,
            content=metrics,
            confidence_score=1.0,
            strategy={},
            agent_sequence=[],
            resource_allocation=metrics,
            quality_gates=[]
        )
    
    def register_agent(self, agent_type: AgentType, agent_instance: Any):
        """Register an agent with the coordinator"""
        self.agent_registry[agent_type] = agent_instance
        logger.info(f"Registered {agent_type.value} agent")
    
    def get_registered_agents(self) -> List[AgentType]:
        """Get list of registered agents"""
        return list(self.agent_registry.keys()) 