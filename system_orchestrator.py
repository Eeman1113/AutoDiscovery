"""
System Orchestrator for the Multi-Agent Autonomous Research System
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from config import Config
from models import (
    AgentType, ResearchSession, ResearchContext, AgentMessage, 
    MessagePriority, ResearchFinding, AnalysisResult, CreativeConcept, 
    ValidationReport
)
from agents.coordinator_agent import CoordinatorAgent
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.creative_agent import CreativeAgent
from agents.validation_agent import ValidationAgent
from agents.documentation_agent import DocumentationAgent

logger = logging.getLogger(__name__)

class SystemOrchestrator:
    """Main orchestrator for the multi-agent research system"""
    
    def __init__(self):
        """Initialize the system orchestrator"""
        self.agents: Dict[AgentType, Any] = {}
        self.active_sessions: Dict[str, ResearchSession] = {}
        self.message_queue: List[AgentMessage] = []
        self.system_status = "initialized"
        self.performance_metrics = {}
        
        # Initialize all agents
        self._initialize_agents()
        
        logger.info("System orchestrator initialized")
    
    def _initialize_agents(self):
        """Initialize all agents in the system"""
        try:
            # Create agent instances
            self.agents[AgentType.COORDINATOR] = CoordinatorAgent()
            self.agents[AgentType.RESEARCH] = ResearchAgent()
            self.agents[AgentType.ANALYSIS] = AnalysisAgent()
            self.agents[AgentType.CREATIVE] = CreativeAgent()
            self.agents[AgentType.VALIDATION] = ValidationAgent()
            self.agents[AgentType.DOCUMENTATION] = DocumentationAgent()
            
            # Register agents with coordinator
            coordinator = self.agents[AgentType.COORDINATOR]
            for agent_type, agent in self.agents.items():
                if agent_type != AgentType.COORDINATOR:
                    coordinator.register_agent(agent_type, agent)
            
            logger.info(f"Initialized {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    async def start_system(self):
        """Start the entire system"""
        try:
            self.system_status = "starting"
            logger.info("Starting multi-agent research system")
            
            # Start all agents
            for agent_type, agent in self.agents.items():
                await agent.start()
                logger.info(f"Started {agent_type.value} agent")
            
            self.system_status = "running"
            logger.info("System started successfully")
            
        except Exception as e:
            self.system_status = "error"
            logger.error(f"Error starting system: {str(e)}")
            raise
    
    async def stop_system(self):
        """Stop the entire system"""
        try:
            self.system_status = "stopping"
            logger.info("Stopping multi-agent research system")
            
            # Stop all agents
            for agent_type, agent in self.agents.items():
                await agent.stop()
                logger.info(f"Stopped {agent_type.value} agent")
            
            self.system_status = "stopped"
            logger.info("System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {str(e)}")
            raise
    
    async def execute_research_query(self, query: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a complete research query using the multi-agent system
        
        Args:
            query: The research query to execute
            options: Optional configuration options
            
        Returns:
            Complete research results
        """
        try:
            session_id = str(uuid.uuid4())
            logger.info(f"Starting research session {session_id} for query: {query}")
            
            # Create research context
            context = ResearchContext(
                query=query,
                domain=options.get("domain") if options else None,
                complexity_score=options.get("complexity_score", 0.5) if options else 0.5,
                research_depth=options.get("research_depth", "comprehensive") if options else "comprehensive",
                output_requirements=options.get("output_requirements", {}) if options else {},
                constraints=options.get("constraints", []) if options else [],
                success_criteria=options.get("success_criteria", []) if options else []
            )
            
            # Create research session
            session = ResearchSession(
                id=session_id,
                context=context,
                status="in_progress"
            )
            
            self.active_sessions[session_id] = session
            
            # Execute research workflow
            result = await self._execute_research_workflow(session)
            
            # Update session status
            session.status = "completed"
            session.completed_at = datetime.now()
            
            # Calculate performance metrics
            self._update_performance_metrics(session_id, result)
            
            logger.info(f"Completed research session {session_id}")
            
            return {
                "session_id": session_id,
                "query": query,
                "status": "completed",
                "results": result,
                "session": session.dict(),
                "performance_metrics": self.performance_metrics.get(session_id, {})
            }
            
        except Exception as e:
            logger.error(f"Error executing research query: {str(e)}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = "error"
            
            return {
                "session_id": session_id if 'session_id' in locals() else None,
                "query": query,
                "status": "error",
                "error": str(e)
            }
    
    async def _execute_research_workflow(self, session: ResearchSession) -> Dict[str, Any]:
        """Execute the complete research workflow"""
        workflow_results = {}
        
        try:
            # Phase 1: Coordination and Strategy
            logger.info(f"Phase 1: Coordination for session {session.id}")
            coordinator_result = await self._execute_coordinator_phase(session)
            workflow_results["coordinator"] = coordinator_result
            
            # Phase 2: Research and Information Gathering
            logger.info(f"Phase 2: Research for session {session.id}")
            research_result = await self._execute_research_phase(session, coordinator_result)
            workflow_results["research"] = research_result
            
            # Phase 3: Analysis and Pattern Recognition
            logger.info(f"Phase 3: Analysis for session {session.id}")
            analysis_result = await self._execute_analysis_phase(session, research_result)
            workflow_results["analysis"] = analysis_result
            
            # Phase 4: Creative Concept Generation
            logger.info(f"Phase 4: Creative for session {session.id}")
            creative_result = await self._execute_creative_phase(session, research_result, analysis_result)
            workflow_results["creative"] = creative_result
            
            # Phase 5: Validation and Quality Assurance
            logger.info(f"Phase 5: Validation for session {session.id}")
            validation_result = await self._execute_validation_phase(session, research_result, analysis_result, creative_result)
            workflow_results["validation"] = validation_result
            
            # Phase 6: Documentation and Final Output
            logger.info(f"Phase 6: Documentation for session {session.id}")
            documentation_result = await self._execute_documentation_phase(session, workflow_results)
            workflow_results["documentation"] = documentation_result
            
            # Update session with results
            session.findings = research_result.get("findings", [])
            session.analysis = analysis_result.get("analysis_result")
            session.creative_concepts = creative_result.get("concepts", [])
            session.validation = validation_result.get("validation_report")
            session.documentation = documentation_result.get("structured_content")
            
            return workflow_results
            
        except Exception as e:
            logger.error(f"Error in research workflow: {str(e)}")
            raise
    
    async def _execute_coordinator_phase(self, session: ResearchSession) -> Dict[str, Any]:
        """Execute the coordinator phase"""
        coordinator = self.agents[AgentType.COORDINATOR]
        
        task_data = {
            "query": session.context.query,
            "session_id": session.id,
            "context": session.context.dict()
        }
        
        result = await coordinator.execute_task(task_data)
        
        if result:
            return result.dict()
        else:
            raise Exception("Coordinator phase failed")
    
    async def _execute_research_phase(self, session: ResearchSession, coordinator_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research phase"""
        research_agent = self.agents[AgentType.RESEARCH]
        
        task_data = {
            "query": session.context.query,
            "context": session.context.dict(),
            "workflow_state": coordinator_result
        }
        
        result = await research_agent.execute_task(task_data)
        
        if result:
            return result.dict()
        else:
            raise Exception("Research phase failed")
    
    async def _execute_analysis_phase(self, session: ResearchSession, research_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the analysis phase"""
        analysis_agent = self.agents[AgentType.ANALYSIS]
        
        task_data = {
            "query": session.context.query,
            "findings": research_result.get("findings", []),
            "sources": research_result.get("sources", []),
            "context": session.context.dict(),
            "workflow_state": {"research_results": research_result}
        }
        
        result = await analysis_agent.execute_task(task_data)
        
        if result:
            return result.dict()
        else:
            raise Exception("Analysis phase failed")
    
    async def _execute_creative_phase(self, session: ResearchSession, research_result: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the creative phase"""
        creative_agent = self.agents[AgentType.CREATIVE]
        
        task_data = {
            "query": session.context.query,
            "findings": research_result.get("findings", []),
            "analysis_results": analysis_result,
            "context": session.context.dict(),
            "workflow_state": {
                "research_results": research_result,
                "analysis_results": analysis_result
            }
        }
        
        result = await creative_agent.execute_task(task_data)
        
        if result:
            return result.dict()
        else:
            raise Exception("Creative phase failed")
    
    async def _execute_validation_phase(self, session: ResearchSession, research_result: Dict[str, Any], analysis_result: Dict[str, Any], creative_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the validation phase"""
        validation_agent = self.agents[AgentType.VALIDATION]
        
        task_data = {
            "query": session.context.query,
            "findings": research_result.get("findings", []),
            "sources": research_result.get("sources", []),
            "analysis_results": analysis_result,
            "creative_concepts": creative_result.get("concepts", []),
            "context": session.context.dict(),
            "workflow_state": {
                "research_results": research_result,
                "analysis_results": analysis_result,
                "creative_results": creative_result
            }
        }
        
        result = await validation_agent.execute_task(task_data)
        
        if result:
            return result.dict()
        else:
            raise Exception("Validation phase failed")
    
    async def _execute_documentation_phase(self, session: ResearchSession, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the documentation phase"""
        documentation_agent = self.agents[AgentType.DOCUMENTATION]
        
        task_data = {
            "query": session.context.query,
            "findings": workflow_results.get("research", {}).get("findings", []),
            "analysis_results": workflow_results.get("analysis", {}),
            "creative_concepts": workflow_results.get("creative", {}).get("concepts", []),
            "validation_report": workflow_results.get("validation", {}).get("validation_report"),
            "context": session.context.dict(),
            "workflow_state": workflow_results
        }
        
        result = await documentation_agent.execute_task(task_data)
        
        if result:
            return result.dict()
        else:
            raise Exception("Documentation phase failed")
    
    def _update_performance_metrics(self, session_id: str, results: Dict[str, Any]):
        """Update performance metrics for a session"""
        metrics = {
            "session_id": session_id,
            "completion_time": datetime.now().isoformat(),
            "phases_completed": len(results),
            "agent_performance": {}
        }
        
        # Calculate agent performance
        for agent_type, result in results.items():
            if result and "confidence_score" in result:
                metrics["agent_performance"][agent_type] = {
                    "confidence_score": result["confidence_score"],
                    "processing_time": result.get("processing_time", 0)
                }
        
        self.performance_metrics[session_id] = metrics
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        agent_statuses = {}
        
        for agent_type, agent in self.agents.items():
            agent_statuses[agent_type.value] = {
                "status": agent.get_status(),
                "performance": agent.get_performance_metrics()
            }
        
        return {
            "system_status": self.system_status,
            "active_sessions": len(self.active_sessions),
            "agents": agent_statuses,
            "total_performance_metrics": len(self.performance_metrics)
        }
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        performance = self.performance_metrics.get(session_id, {})
        
        return {
            "session_id": session_id,
            "session": session.dict(),
            "performance_metrics": performance
        }
    
    async def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        sessions = []
        
        for session_id, session in self.active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "query": session.context.query,
                "status": session.status,
                "created_at": session.created_at.isoformat(),
                "completed_at": session.completed_at.isoformat() if session.completed_at else None
            })
        
        return sessions
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an active session"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.status = "cancelled"
        
        logger.info(f"Cancelled session {session_id}")
        return True
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old completed sessions"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session in self.active_sessions.items():
            if session.completed_at:
                age_hours = (current_time - session.completed_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            if session_id in self.performance_metrics:
                del self.performance_metrics[session_id]
        
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
    
    async def send_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Send a message to a specific agent"""
        if message.recipient not in self.agents:
            logger.error(f"Agent {message.recipient.value} not found")
            return None
        
        agent = self.agents[message.recipient]
        result = await agent.process_message(message)
        
        if result:
            return result.dict()
        else:
            return None
    
    def get_agent_info(self, agent_type: AgentType) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent"""
        if agent_type not in self.agents:
            return None
        
        agent = self.agents[agent_type]
        return {
            "agent_type": agent_type.value,
            "status": agent.get_status(),
            "performance": agent.get_performance_metrics()
        } 