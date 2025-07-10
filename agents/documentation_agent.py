"""
Documentation Agent for the Multi-Agent Autonomous Research System
"""
import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json

from base_agent import BaseAgent
from models import (
    AgentType, AgentMessage, DocumentationOutput, ResearchFinding, 
    ResearchSource, AnalysisResult, CreativeConcept, ValidationReport
)
from config import Config

logger = logging.getLogger(__name__)

class DocumentationAgent(BaseAgent):
    """Documentation creation and formatting specialist"""
    
    def __init__(self):
        """Initialize the documentation agent"""
        super().__init__(AgentType.DOCUMENTATION)
        self.documentation_templates = self._load_documentation_templates()
        self.format_options = self._load_format_options()
        
    def _load_documentation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined documentation templates"""
        return {
            "academic": {
                "structure": [
                    "title",
                    "abstract",
                    "introduction",
                    "literature_review",
                    "methodology",
                    "findings",
                    "analysis",
                    "discussion",
                    "conclusion",
                    "references"
                ],
                "style": "formal",
                "citation_style": "academic",
                "include_visualizations": True
            },
            "executive_summary": {
                "structure": [
                    "title",
                    "executive_summary",
                    "key_findings",
                    "recommendations",
                    "next_steps",
                    "appendix"
                ],
                "style": "business",
                "citation_style": "minimal",
                "include_visualizations": True
            },
            "technical_spec": {
                "structure": [
                    "title",
                    "overview",
                    "requirements",
                    "specifications",
                    "implementation",
                    "testing",
                    "deployment",
                    "references"
                ],
                "style": "technical",
                "citation_style": "technical",
                "include_visualizations": True
            },
            "creative_report": {
                "structure": [
                    "title",
                    "concept_overview",
                    "creative_process",
                    "innovations",
                    "feasibility_analysis",
                    "implementation_roadmap",
                    "conclusion"
                ],
                "style": "creative",
                "citation_style": "flexible",
                "include_visualizations": True
            }
        }
    
    def _load_format_options(self) -> Dict[str, Dict[str, Any]]:
        """Load format options for different output types"""
        return {
            "markdown": {
                "extension": ".md",
                "features": ["headers", "lists", "links", "images", "tables", "code_blocks"],
                "compatibility": "universal"
            },
            "html": {
                "extension": ".html",
                "features": ["styling", "interactive", "responsive", "embedding"],
                "compatibility": "web"
            },
            "pdf": {
                "extension": ".pdf",
                "features": ["printable", "formatted", "secure"],
                "compatibility": "print"
            },
            "json": {
                "extension": ".json",
                "features": ["structured", "machine_readable", "api_friendly"],
                "compatibility": "programmatic"
            }
        }
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Optional[DocumentationOutput]:
        """Execute a documentation task"""
        query = task_data.get("query", "")
        findings = task_data.get("findings", [])
        analysis_results = task_data.get("analysis_results", {})
        creative_concepts = task_data.get("creative_concepts", [])
        validation_report = task_data.get("validation_report", {})
        context = task_data.get("context", {})
        
        if not query:
            logger.error("No query provided for documentation task")
            return None
        
        self.log_activity("Starting documentation task", {"query": query})
        
        # Determine documentation approach
        doc_approach = self._determine_documentation_approach(context, findings, creative_concepts)
        
        # Create structured content
        structured_content = await self._create_structured_content(
            query, findings, analysis_results, creative_concepts, validation_report, doc_approach
        )
        
        # Generate visualizations
        visualizations = await self._generate_visualizations(findings, analysis_results, creative_concepts)
        
        # Create citations
        citations = self._create_citations(findings, analysis_results)
        
        # Generate format options
        format_options = self._generate_format_options(structured_content, visualizations, citations)
        
        return DocumentationOutput(
            agent_type=AgentType.DOCUMENTATION,
            content={
                "structured_content": structured_content,
                "visualizations": visualizations,
                "citations": citations
            },
            confidence_score=self._calculate_documentation_confidence(structured_content, visualizations),
            structured_content=structured_content,
            format_options=format_options,
            visualizations=visualizations,
            citations=citations
        )
    
    async def execute_query(self, query_data: Dict[str, Any]) -> Optional[DocumentationOutput]:
        """Execute a documentation query"""
        query_type = query_data.get("type", "documentation")
        
        if query_type == "format_conversion":
            return await self._convert_format(query_data)
        elif query_type == "content_summary":
            return await self._create_summary(query_data)
        elif query_type == "visualization_request":
            return await self._create_visualization(query_data)
        else:
            logger.warning(f"Unknown documentation query type: {query_type}")
            return None
    
    def _determine_documentation_approach(
        self, 
        context: Dict[str, Any], 
        findings: List[Any], 
        creative_concepts: List[Any]
    ) -> Dict[str, Any]:
        """Determine the appropriate documentation approach"""
        domain = context.get("domain", "general")
        complexity_score = context.get("complexity_score", 0.5)
        has_creative_content = len(creative_concepts) > 0
        
        # Select base template
        if has_creative_content:
            base_template = self.documentation_templates["creative_report"]
        elif domain in ["academic", "scientific"]:
            base_template = self.documentation_templates["academic"]
        elif domain in ["business", "financial"]:
            base_template = self.documentation_templates["executive_summary"]
        elif domain in ["technology", "engineering"]:
            base_template = self.documentation_templates["technical_spec"]
        else:
            base_template = self.documentation_templates["executive_summary"]
        
        # Customize based on complexity
        if complexity_score > 0.7:
            base_template["include_detailed_analysis"] = True
            base_template["include_methodology"] = True
        else:
            base_template["include_detailed_analysis"] = False
            base_template["include_methodology"] = False
        
        return base_template
    
    async def _create_structured_content(
        self, 
        query: str, 
        findings: List[Any], 
        analysis_results: Dict[str, Any],
        creative_concepts: List[Any], 
        validation_report: Dict[str, Any],
        approach: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create structured content based on the approach"""
        self.log_activity("Creating structured content", {"approach": approach.get("style", "standard")})
        
        structured_content = {}
        structure = approach.get("structure", [])
        
        # Generate content for each section
        for section in structure:
            section_content = await self._generate_section_content(
                section, query, findings, analysis_results, creative_concepts, validation_report, approach
            )
            structured_content[section] = section_content
        
        # Add metadata
        structured_content["metadata"] = {
            "query": query,
            "generated_at": datetime.now().isoformat(),
            "template": approach.get("style", "standard"),
            "sections": structure,
            "findings_count": len(findings),
            "concepts_count": len(creative_concepts)
        }
        
        return structured_content
    
    async def _generate_section_content(
        self, 
        section: str, 
        query: str, 
        findings: List[Any], 
        analysis_results: Dict[str, Any],
        creative_concepts: List[Any], 
        validation_report: Dict[str, Any],
        approach: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate content for a specific section"""
        prompt = f"""
        Generate content for the '{section}' section of a {approach.get('style', 'standard')} document.
        
        Query: {query}
        Findings: {len(findings)} findings
        Creative concepts: {len(creative_concepts)} concepts
        Analysis results: {len(analysis_results)} analysis components
        
        Create comprehensive, well-structured content for this section.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            return {
                "content": response["content"],
                "generated_at": datetime.now().isoformat(),
                "confidence": response.get("confidence_score", 0.8)
            }
        else:
            return {
                "content": f"Content for {section} section",
                "generated_at": datetime.now().isoformat(),
                "confidence": 0.5
            }
    
    async def _generate_visualizations(
        self, 
        findings: List[Any], 
        analysis_results: Dict[str, Any], 
        creative_concepts: List[Any]
    ) -> List[Dict[str, Any]]:
        """Generate visualizations for the documentation"""
        visualizations = []
        
        # Generate findings summary visualization
        if findings:
            findings_viz = await self._create_findings_visualization(findings)
            visualizations.append(findings_viz)
        
        # Generate analysis visualization
        if analysis_results:
            analysis_viz = await self._create_analysis_visualization(analysis_results)
            visualizations.append(analysis_viz)
        
        # Generate creative concepts visualization
        if creative_concepts:
            concepts_viz = await self._create_concepts_visualization(creative_concepts)
            visualizations.append(concepts_viz)
        
        return visualizations
    
    async def _create_findings_visualization(self, findings: List[Any]) -> Dict[str, Any]:
        """Create visualization for research findings"""
        prompt = f"""
        Create a visualization description for {len(findings)} research findings.
        
        Findings: {[f.title if hasattr(f, 'title') else str(f) for f in findings]}
        
        Suggest an appropriate visualization type and structure.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        return {
            "type": "findings_summary",
            "title": "Research Findings Summary",
            "description": response.get("content", "Visualization of research findings"),
            "data": {
                "findings_count": len(findings),
                "categories": self._extract_finding_categories(findings),
                "confidence_scores": [f.confidence_score if hasattr(f, 'confidence_score') else 0.5 for f in findings]
            },
            "visualization_type": "bar_chart",
            "generated_at": datetime.now().isoformat()
        }
    
    async def _create_analysis_visualization(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization for analysis results"""
        prompt = f"""
        Create a visualization description for analysis results.
        
        Analysis components: {list(analysis_results.keys())}
        
        Suggest appropriate visualization types for different analysis components.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        return {
            "type": "analysis_summary",
            "title": "Analysis Results",
            "description": response.get("content", "Visualization of analysis results"),
            "data": {
                "analysis_components": list(analysis_results.keys()),
                "component_count": len(analysis_results)
            },
            "visualization_type": "dashboard",
            "generated_at": datetime.now().isoformat()
        }
    
    async def _create_concepts_visualization(self, creative_concepts: List[Any]) -> Dict[str, Any]:
        """Create visualization for creative concepts"""
        prompt = f"""
        Create a visualization description for {len(creative_concepts)} creative concepts.
        
        Concepts: {[c.title if hasattr(c, 'title') else str(c) for c in creative_concepts]}
        
        Suggest an appropriate visualization type for creative concepts.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        return {
            "type": "concepts_overview",
            "title": "Creative Concepts Overview",
            "description": response.get("content", "Visualization of creative concepts"),
            "data": {
                "concepts_count": len(creative_concepts),
                "innovation_scores": [c.innovation_score if hasattr(c, 'innovation_score') else 0.5 for c in creative_concepts],
                "feasibility_scores": [c.feasibility_score if hasattr(c, 'feasibility_score') else 0.5 for c in creative_concepts]
            },
            "visualization_type": "scatter_plot",
            "generated_at": datetime.now().isoformat()
        }
    
    def _create_citations(self, findings: List[Any], analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create citations for the documentation"""
        citations = []
        
        # Create citations for findings
        for i, finding in enumerate(findings):
            if hasattr(finding, 'sources') and finding.sources:
                for source in finding.sources:
                    citation = {
                        "id": f"citation_{i}_{source.id}",
                        "type": "source",
                        "title": source.title,
                        "author": source.author,
                        "url": source.url,
                        "publication_date": source.publication_date.isoformat() if source.publication_date else None,
                        "quality_score": source.quality_score,
                        "referenced_in": finding.title if hasattr(finding, 'title') else f"Finding {i+1}"
                    }
                    citations.append(citation)
        
        # Create citations for analysis
        if analysis_results:
            analysis_citation = {
                "id": "citation_analysis",
                "type": "analysis",
                "title": "Research Analysis",
                "description": "Comprehensive analysis of research findings",
                "generated_at": datetime.now().isoformat(),
                "confidence_score": analysis_results.get("confidence_score", 0.8)
            }
            citations.append(analysis_citation)
        
        return citations
    
    def _generate_format_options(
        self, 
        structured_content: Dict[str, Any], 
        visualizations: List[Dict[str, Any]], 
        citations: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate different format options for the documentation"""
        format_options = {}
        
        for format_name, format_config in self.format_options.items():
            if format_name == "markdown":
                format_options["markdown"] = self._generate_markdown(structured_content, visualizations, citations)
            elif format_name == "html":
                format_options["html"] = self._generate_html(structured_content, visualizations, citations)
            elif format_name == "json":
                format_options["json"] = self._generate_json(structured_content, visualizations, citations)
            elif format_name == "pdf":
                format_options["pdf"] = "PDF generation requires additional processing"
        
        return format_options
    
    def _generate_markdown(
        self, 
        structured_content: Dict[str, Any], 
        visualizations: List[Dict[str, Any]], 
        citations: List[Dict[str, Any]]
    ) -> str:
        """Generate markdown format"""
        markdown_content = []
        
        # Add title
        if "title" in structured_content:
            markdown_content.append(f"# {structured_content['title']['content']}")
            markdown_content.append("")
        
        # Add sections
        for section_name, section_content in structured_content.items():
            if section_name in ["title", "metadata"]:
                continue
            
            markdown_content.append(f"## {section_name.replace('_', ' ').title()}")
            markdown_content.append("")
            markdown_content.append(section_content.get("content", ""))
            markdown_content.append("")
        
        # Add visualizations
        if visualizations:
            markdown_content.append("## Visualizations")
            markdown_content.append("")
            for viz in visualizations:
                markdown_content.append(f"### {viz['title']}")
                markdown_content.append(viz['description'])
                markdown_content.append("")
        
        # Add citations
        if citations:
            markdown_content.append("## References")
            markdown_content.append("")
            for citation in citations:
                if citation.get("title"):
                    markdown_content.append(f"- {citation['title']}")
                    if citation.get("author"):
                        markdown_content.append(f"  - Author: {citation['author']}")
                    if citation.get("url"):
                        markdown_content.append(f"  - URL: {citation['url']}")
                    markdown_content.append("")
        
        return "\n".join(markdown_content)
    
    def _generate_html(
        self, 
        structured_content: Dict[str, Any], 
        visualizations: List[Dict[str, Any]], 
        citations: List[Dict[str, Any]]
    ) -> str:
        """Generate HTML format"""
        html_content = []
        
        # HTML header
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html lang='en'>")
        html_content.append("<head>")
        html_content.append("    <meta charset='UTF-8'>")
        html_content.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html_content.append("    <title>Research Documentation</title>")
        html_content.append("    <style>")
        html_content.append("        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }")
        html_content.append("        h1, h2, h3 { color: #333; }")
        html_content.append("        .section { margin-bottom: 30px; }")
        html_content.append("        .visualization { border: 1px solid #ddd; padding: 20px; margin: 20px 0; }")
        html_content.append("        .citation { margin: 10px 0; padding: 10px; background: #f9f9f9; }")
        html_content.append("    </style>")
        html_content.append("</head>")
        html_content.append("<body>")
        
        # Add content
        for section_name, section_content in structured_content.items():
            if section_name in ["metadata"]:
                continue
            
            html_content.append(f"<div class='section'>")
            if section_name == "title":
                html_content.append(f"<h1>{section_content.get('content', 'Research Documentation')}</h1>")
            else:
                html_content.append(f"<h2>{section_name.replace('_', ' ').title()}</h2>")
                html_content.append(f"<p>{section_content.get('content', '')}</p>")
            html_content.append("</div>")
        
        # Add visualizations
        if visualizations:
            html_content.append("<div class='section'>")
            html_content.append("<h2>Visualizations</h2>")
            for viz in visualizations:
                html_content.append("<div class='visualization'>")
                html_content.append(f"<h3>{viz['title']}</h3>")
                html_content.append(f"<p>{viz['description']}</p>")
                html_content.append("</div>")
            html_content.append("</div>")
        
        # Add citations
        if citations:
            html_content.append("<div class='section'>")
            html_content.append("<h2>References</h2>")
            for citation in citations:
                html_content.append("<div class='citation'>")
                if citation.get("title"):
                    html_content.append(f"<strong>{citation['title']}</strong>")
                if citation.get("author"):
                    html_content.append(f"<br>Author: {citation['author']}")
                if citation.get("url"):
                    html_content.append(f"<br>URL: <a href='{citation['url']}'>{citation['url']}</a>")
                html_content.append("</div>")
            html_content.append("</div>")
        
        # HTML footer
        html_content.append("</body>")
        html_content.append("</html>")
        
        return "\n".join(html_content)
    
    def _generate_json(
        self, 
        structured_content: Dict[str, Any], 
        visualizations: List[Dict[str, Any]], 
        citations: List[Dict[str, Any]]
    ) -> str:
        """Generate JSON format"""
        json_data = {
            "documentation": {
                "content": structured_content,
                "visualizations": visualizations,
                "citations": citations,
                "generated_at": datetime.now().isoformat(),
                "format": "json"
            }
        }
        
        return json.dumps(json_data, indent=2, default=str)
    
    def _calculate_documentation_confidence(
        self, 
        structured_content: Dict[str, Any], 
        visualizations: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for documentation"""
        confidence = 0.7  # Base confidence
        
        # Content quality
        if structured_content:
            confidence += 0.1
        
        # Visualization quality
        if visualizations:
            confidence += 0.1
        
        # Section completeness
        required_sections = ["title", "introduction", "conclusion"]
        present_sections = sum(1 for section in required_sections if section in structured_content)
        section_score = present_sections / len(required_sections)
        confidence += section_score * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_finding_categories(self, findings: List[Any]) -> List[str]:
        """Extract categories from findings"""
        categories = []
        
        for finding in findings:
            if hasattr(finding, 'category') and finding.category:
                categories.append(finding.category)
            elif hasattr(finding, 'tags') and finding.tags:
                categories.extend(finding.tags)
        
        return list(set(categories))  # Remove duplicates
    
    # Query-specific methods
    async def _convert_format(self, query_data: Dict[str, Any]) -> Optional[DocumentationOutput]:
        """Convert documentation to a different format"""
        content = query_data.get("content", "")
        target_format = query_data.get("target_format", "markdown")
        
        if not content:
            return None
        
        # Simplified format conversion
        if target_format == "markdown":
            converted_content = f"# Converted Content\n\n{content}"
        elif target_format == "html":
            converted_content = f"<h1>Converted Content</h1><p>{content}</p>"
        else:
            converted_content = content
        
        return DocumentationOutput(
            agent_type=AgentType.DOCUMENTATION,
            content={"converted_content": converted_content},
            confidence_score=0.8,
            structured_content={"converted": {"content": converted_content}},
            format_options={target_format: converted_content},
            visualizations=[],
            citations=[]
        )
    
    async def _create_summary(self, query_data: Dict[str, Any]) -> Optional[DocumentationOutput]:
        """Create a summary of the documentation"""
        content = query_data.get("content", "")
        
        if not content:
            return None
        
        prompt = f"""
        Create a concise summary of this content:
        
        {content}
        
        Provide a 2-3 paragraph summary that captures the key points.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        summary_content = response.get("content", "Summary not available") if response.get("success") else "Summary not available"
        
        return DocumentationOutput(
            agent_type=AgentType.DOCUMENTATION,
            content={"summary": summary_content},
            confidence_score=0.8,
            structured_content={"summary": {"content": summary_content}},
            format_options={"summary": summary_content},
            visualizations=[],
            citations=[]
        )
    
    async def _create_visualization(self, query_data: Dict[str, Any]) -> Optional[DocumentationOutput]:
        """Create a specific visualization"""
        data = query_data.get("data", {})
        visualization_type = query_data.get("type", "chart")
        
        if not data:
            return None
        
        visualization = {
            "type": visualization_type,
            "title": f"{visualization_type.title()} Visualization",
            "description": f"Visualization of {visualization_type} data",
            "data": data,
            "generated_at": datetime.now().isoformat()
        }
        
        return DocumentationOutput(
            agent_type=AgentType.DOCUMENTATION,
            content={"visualization": visualization},
            confidence_score=0.8,
            structured_content={"visualization": {"content": str(visualization)}},
            format_options={"visualization": str(visualization)},
            visualizations=[visualization],
            citations=[]
        ) 