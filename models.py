"""
Data models for the Multi-Agent Autonomous Research System
"""
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid

class AgentType(str, Enum):
    """Enumeration of agent types"""
    COORDINATOR = "coordinator"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    VALIDATION = "validation"
    DOCUMENTATION = "documentation"

class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class SourceQuality(str, Enum):
    """Source quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ACADEMIC = "academic"
    EXPERT = "expert"

class ValidationLayer(str, Enum):
    """Validation layer types"""
    FACTUAL = "factual"
    LOGICAL = "logical"
    SOURCE = "source"
    BIAS = "bias"
    COMPLETENESS = "completeness"

class ResearchSource(BaseModel):
    """Model for research sources"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: Optional[str] = None
    title: str
    content: str
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_level: SourceQuality
    domain: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)

class ResearchFinding(BaseModel):
    """Model for individual research findings"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    sources: List[ResearchSource] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

class AnalysisResult(BaseModel):
    """Model for analysis results"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patterns: List[Dict[str, Any]] = Field(default_factory=list)
    correlations: List[Dict[str, Any]] = Field(default_factory=list)
    trends: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    confidence_intervals: Dict[str, List[float]] = Field(default_factory=dict)
    statistical_metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class CreativeConcept(BaseModel):
    """Model for creative concepts and solutions"""
    id: str = str(uuid.uuid4())
    title: str
    description: str
    innovation_score: float = Field(ge=0.0, le=1.0)
    feasibility_score: float = Field(ge=0.0, le=1.0)
    implementation_path: List[str] = Field(default_factory=list)
    alternatives: List[str] = Field(default_factory=list)
    inspiration_sources: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

class ValidationReport(BaseModel):
    """Model for validation reports"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    layers: Dict[ValidationLayer, Dict[str, Any]] = Field(default_factory=dict)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class AgentMessage(BaseModel):
    """Model for inter-agent communication"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: AgentType
    recipient: AgentType
    message_type: str
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_references: List[str] = Field(default_factory=list)

class ResearchContext(BaseModel):
    """Model for shared research context"""
    query: str
    domain: Optional[str] = None
    complexity_score: float = Field(ge=0.0, le=1.0)
    research_depth: str = "comprehensive"
    output_requirements: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

class ResearchSession(BaseModel):
    """Model for complete research session"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context: ResearchContext
    findings: List[ResearchFinding] = Field(default_factory=list)
    analysis: Optional[AnalysisResult] = None
    creative_concepts: List[CreativeConcept] = Field(default_factory=list)
    validation: Optional[ValidationReport] = None
    documentation: Optional[Dict[str, Any]] = None
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    status: str = "in_progress"
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

class AgentOutput(BaseModel):
    """Base model for agent outputs"""
    agent_type: AgentType
    content: Dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: Optional[float] = None

class CoordinatorOutput(AgentOutput):
    """Coordinator agent output"""
    strategy: Dict[str, Any]
    agent_sequence: List[AgentType]
    resource_allocation: Dict[str, Any]
    quality_gates: List[Dict[str, Any]]

class ResearchOutput(AgentOutput):
    """Research agent output"""
    sources: List[ResearchSource]
    findings: List[ResearchFinding]
    search_strategy: Dict[str, Any]
    coverage_analysis: Dict[str, Any]

class AnalysisOutput(AgentOutput):
    """Analysis agent output"""
    analysis_result: AnalysisResult
    insights: List[str]
    correlations: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]

class CreativeOutput(AgentOutput):
    """Creative agent output"""
    concepts: List[CreativeConcept]
    innovation_metrics: Dict[str, float]
    synthesis_approach: str
    cross_domain_insights: List[str]

class ValidationOutput(AgentOutput):
    """Validation agent output"""
    validation_report: ValidationReport
    quality_assessment: Dict[str, float]
    improvement_suggestions: List[str]
    risk_assessment: Dict[str, Any]

class DocumentationOutput(AgentOutput):
    """Documentation agent output"""
    structured_content: Dict[str, Any]
    format_options: Dict[str, str]
    visualizations: List[Dict[str, Any]]
    citations: List[Dict[str, Any]] 