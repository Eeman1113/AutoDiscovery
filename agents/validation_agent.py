"""
Validation Agent for the Multi-Agent Autonomous Research System
"""
import asyncio
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict

from base_agent import BaseAgent
from models import (
    AgentType, AgentMessage, ValidationOutput, ValidationReport, 
    ValidationLayer, ResearchFinding, ResearchSource
)
from config import Config

logger = logging.getLogger(__name__)

class ValidationAgent(BaseAgent):
    """Comprehensive validation and quality assurance specialist"""
    
    def __init__(self):
        """Initialize the validation agent"""
        super().__init__(AgentType.VALIDATION)
        self.validation_methods = self._load_validation_methods()
        self.quality_metrics = self._load_quality_metrics()
        
    def _load_validation_methods(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined validation methods"""
        return {
            "factual": {
                "source_verification": True,
                "cross_reference_check": True,
                "expert_opinion_validation": True,
                "statistical_validation": True
            },
            "logical": {
                "consistency_check": True,
                "contradiction_detection": True,
                "logical_flow_analysis": True,
                "assumption_validation": True
            },
            "source": {
                "credibility_assessment": True,
                "authority_verification": True,
                "recency_check": True,
                "bias_detection": True
            },
            "bias": {
                "perspective_analysis": True,
                "conflict_of_interest_check": True,
                "language_bias_detection": True,
                "cultural_bias_assessment": True
            },
            "completeness": {
                "coverage_analysis": True,
                "gap_identification": True,
                "comprehensiveness_check": True,
                "depth_assessment": True
            }
        }
    
    def _load_quality_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Load quality metrics definitions"""
        return {
            "accuracy": {
                "weight": 0.3,
                "threshold": 0.8,
                "description": "Factual accuracy of information"
            },
            "reliability": {
                "weight": 0.25,
                "threshold": 0.7,
                "description": "Reliability of sources and methods"
            },
            "completeness": {
                "weight": 0.2,
                "threshold": 0.6,
                "description": "Completeness of coverage"
            },
            "consistency": {
                "weight": 0.15,
                "threshold": 0.8,
                "description": "Internal consistency of findings"
            },
            "objectivity": {
                "weight": 0.1,
                "threshold": 0.7,
                "description": "Objectivity and lack of bias"
            }
        }
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Optional[ValidationOutput]:
        """Execute a validation task"""
        findings = task_data.get("findings", [])
        sources = task_data.get("sources", [])
        analysis_results = task_data.get("analysis_results", {})
        context = task_data.get("context", {})
        
        if not findings and not sources:
            logger.error("No findings or sources provided for validation")
            return None
        
        self.log_activity("Starting validation task", {"findings_count": len(findings), "sources_count": len(sources)})
        
        # Determine validation approach
        validation_approach = self._determine_validation_approach(context, findings, sources)
        
        # Perform comprehensive validation
        validation_result = await self._perform_comprehensive_validation(
            findings, sources, analysis_results, validation_approach
        )
        
        # Generate quality assessment
        quality_assessment = self._generate_quality_assessment(validation_result)
        
        # Generate improvement suggestions
        improvement_suggestions = await self._generate_improvement_suggestions(validation_result)
        
        # Perform risk assessment
        risk_assessment = self._perform_risk_assessment(validation_result)
        
        return ValidationOutput(
            agent_type=AgentType.VALIDATION,
            content=validation_result,
            confidence_score=validation_result.get("overall_confidence", 0.8),
            validation_report=validation_result,
            quality_assessment=quality_assessment,
            improvement_suggestions=improvement_suggestions,
            risk_assessment=risk_assessment
        )
    
    async def execute_query(self, query_data: Dict[str, Any]) -> Optional[ValidationOutput]:
        """Execute a validation query"""
        query_type = query_data.get("type", "validation")
        
        if query_type == "fact_check":
            return await self._fact_check_content(query_data)
        elif query_type == "source_validation":
            return await self._validate_sources(query_data)
        elif query_type == "bias_analysis":
            return await self._analyze_bias(query_data)
        else:
            logger.warning(f"Unknown validation query type: {query_type}")
            return None
    
    def _determine_validation_approach(
        self, 
        context: Dict[str, Any], 
        findings: List[Any], 
        sources: List[Any]
    ) -> Dict[str, Any]:
        """Determine the appropriate validation approach"""
        domain = context.get("domain", "general")
        complexity_score = context.get("complexity_score", 0.5)
        
        # Select base approach
        if complexity_score > 0.7:
            base_approach = {
                "factual": True,
                "logical": True,
                "source": True,
                "bias": True,
                "completeness": True
            }
        elif complexity_score > 0.4:
            base_approach = {
                "factual": True,
                "logical": True,
                "source": True,
                "bias": False,
                "completeness": True
            }
        else:
            base_approach = {
                "factual": True,
                "logical": True,
                "source": True,
                "bias": False,
                "completeness": False
            }
        
        # Customize based on domain
        if domain in ["scientific", "academic"]:
            base_approach["factual"] = True
            base_approach["source"] = True
        elif domain in ["business", "financial"]:
            base_approach["bias"] = True
            base_approach["logical"] = True
        
        return base_approach
    
    async def _perform_comprehensive_validation(
        self, 
        findings: List[Any], 
        sources: List[Any], 
        analysis_results: Dict[str, Any],
        approach: Dict[str, Any]
    ) -> ValidationReport:
        """Perform comprehensive validation using multiple methods"""
        self.log_activity("Performing comprehensive validation", {"approach": approach})
        
        validation_layers = {}
        
        # Perform validation for each enabled layer
        if approach.get("factual"):
            factual_validation = await self._perform_factual_validation(findings, sources)
            validation_layers[ValidationLayer.FACTUAL] = factual_validation
        
        if approach.get("logical"):
            logical_validation = await self._perform_logical_validation(findings, analysis_results)
            validation_layers[ValidationLayer.LOGICAL] = logical_validation
        
        if approach.get("source"):
            source_validation = await self._perform_source_validation(sources)
            validation_layers[ValidationLayer.SOURCE] = source_validation
        
        if approach.get("bias"):
            bias_validation = await self._perform_bias_validation(findings, sources)
            validation_layers[ValidationLayer.BIAS] = bias_validation
        
        if approach.get("completeness"):
            completeness_validation = await self._perform_completeness_validation(findings, sources)
            validation_layers[ValidationLayer.COMPLETENESS] = completeness_validation
        
        # Identify contradictions
        contradictions = self._identify_contradictions(validation_layers, findings)
        
        # Identify gaps
        gaps = self._identify_gaps(validation_layers, findings)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(validation_layers, contradictions, gaps)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(validation_layers)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(validation_layers)
        
        return ValidationReport(
            layers=validation_layers,
            overall_confidence=overall_confidence,
            contradictions=contradictions,
            gaps=gaps,
            recommendations=recommendations,
            quality_metrics=quality_metrics
        )
    
    async def _perform_factual_validation(self, findings: List[Any], sources: List[Any]) -> Dict[str, Any]:
        """Perform factual validation"""
        prompt = f"""
        Perform factual validation on the following research data:
        
        Findings: {[f.title if hasattr(f, 'title') else str(f) for f in findings]}
        Sources: {len(sources)} sources
        
        Validate:
        1. Factual accuracy of claims
        2. Source verification
        3. Cross-reference consistency
        4. Statistical validity
        5. Expert opinion alignment
        
        Provide validation results with confidence scores.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            validation_result = self._extract_validation_result(response["content"], "factual")
            return validation_result
        else:
            return {"status": "failed", "confidence": 0.0, "issues": ["Validation failed"]}
    
    async def _perform_logical_validation(self, findings: List[Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform logical validation"""
        prompt = f"""
        Perform logical validation on the research findings:
        
        Findings: {[f.title if hasattr(f, 'title') else str(f) for f in findings]}
        Analysis results: {analysis_results}
        
        Validate:
        1. Logical consistency
        2. Contradiction detection
        3. Logical flow analysis
        4. Assumption validation
        5. Reasoning soundness
        
        Identify any logical issues or inconsistencies.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            validation_result = self._extract_validation_result(response["content"], "logical")
            return validation_result
        else:
            return {"status": "failed", "confidence": 0.0, "issues": ["Validation failed"]}
    
    async def _perform_source_validation(self, sources: List[Any]) -> Dict[str, Any]:
        """Perform source validation"""
        prompt = f"""
        Validate the credibility and reliability of these sources:
        
        Sources: {len(sources)} sources
        
        Assess:
        1. Source credibility
        2. Authority verification
        3. Recency and relevance
        4. Bias detection
        5. Quality indicators
        
        Provide source validation results.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            validation_result = self._extract_validation_result(response["content"], "source")
            return validation_result
        else:
            return {"status": "failed", "confidence": 0.0, "issues": ["Validation failed"]}
    
    async def _perform_bias_validation(self, findings: List[Any], sources: List[Any]) -> Dict[str, Any]:
        """Perform bias validation"""
        prompt = f"""
        Analyze potential biases in the research:
        
        Findings: {[f.title if hasattr(f, 'title') else str(f) for f in findings]}
        Sources: {len(sources)} sources
        
        Detect:
        1. Perspective bias
        2. Conflict of interest
        3. Language bias
        4. Cultural bias
        5. Confirmation bias
        
        Identify any biases and their potential impact.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            validation_result = self._extract_validation_result(response["content"], "bias")
            return validation_result
        else:
            return {"status": "failed", "confidence": 0.0, "issues": ["Validation failed"]}
    
    async def _perform_completeness_validation(self, findings: List[Any], sources: List[Any]) -> Dict[str, Any]:
        """Perform completeness validation"""
        prompt = f"""
        Assess the completeness of the research:
        
        Findings: {[f.title if hasattr(f, 'title') else str(f) for f in findings]}
        Sources: {len(sources)} sources
        
        Evaluate:
        1. Coverage comprehensiveness
        2. Gap identification
        3. Depth of analysis
        4. Missing perspectives
        5. Completeness indicators
        
        Identify any gaps or missing elements.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            validation_result = self._extract_validation_result(response["content"], "completeness")
            return validation_result
        else:
            return {"status": "failed", "confidence": 0.0, "issues": ["Validation failed"]}
    
    def _identify_contradictions(self, validation_layers: Dict[ValidationLayer, Dict[str, Any]], findings: List[Any]) -> List[Dict[str, Any]]:
        """Identify contradictions across validation layers"""
        contradictions = []
        
        # Compare findings for contradictions
        if len(findings) > 1:
            for i, finding1 in enumerate(findings):
                for finding2 in findings[i+1:]:
                    contradiction = self._check_finding_contradiction(finding1, finding2)
                    if contradiction:
                        contradictions.append(contradiction)
        
        # Check for validation layer contradictions
        layer_issues = defaultdict(list)
        for layer, validation in validation_layers.items():
            if validation.get("issues"):
                layer_issues[layer.value].extend(validation["issues"])
        
        # Look for conflicting issues
        for layer, issues in layer_issues.items():
            for issue in issues:
                if "contradict" in issue.lower() or "conflict" in issue.lower():
                    contradictions.append({
                        "type": "validation_contradiction",
                        "layer": layer,
                        "description": issue,
                        "severity": "moderate"
                    })
        
        return contradictions
    
    def _check_finding_contradiction(self, finding1: Any, finding2: Any) -> Optional[Dict[str, Any]]:
        """Check for contradiction between two findings"""
        # Simplified contradiction detection
        # In production, use more sophisticated NLP techniques
        
        title1 = finding1.title if hasattr(finding1, 'title') else str(finding1)
        title2 = finding2.title if hasattr(finding2, 'title') else str(finding2)
        
        # Look for opposite keywords
        opposite_pairs = [
            ("increase", "decrease"),
            ("positive", "negative"),
            ("success", "failure"),
            ("benefit", "harm"),
            ("support", "oppose")
        ]
        
        for word1, word2 in opposite_pairs:
            if word1 in title1.lower() and word2 in title2.lower():
                return {
                    "type": "finding_contradiction",
                    "finding1": title1,
                    "finding2": title2,
                    "description": f"Contradictory findings: {word1} vs {word2}",
                    "severity": "high"
                }
        
        return None
    
    def _identify_gaps(self, validation_layers: Dict[ValidationLayer, Dict[str, Any]], findings: List[Any]) -> List[str]:
        """Identify gaps in the research"""
        gaps = []
        
        # Check completeness layer for gaps
        if ValidationLayer.COMPLETENESS in validation_layers:
            completeness_validation = validation_layers[ValidationLayer.COMPLETENESS]
            if completeness_validation.get("issues"):
                gaps.extend(completeness_validation["issues"])
        
        # Check for missing perspectives
        if len(findings) < 3:
            gaps.append("Limited number of findings - may miss important perspectives")
        
        # Check for domain coverage
        domains = set()
        for finding in findings:
            if hasattr(finding, 'category') and finding.category:
                domains.add(finding.category)
        
        if len(domains) < 2:
            gaps.append("Limited domain coverage - may miss cross-domain insights")
        
        return gaps
    
    async def _generate_recommendations(
        self, 
        validation_layers: Dict[ValidationLayer, Dict[str, Any]], 
        contradictions: List[Dict[str, Any]], 
        gaps: List[str]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Generate recommendations based on validation issues
        for layer, validation in validation_layers.items():
            if validation.get("confidence", 1.0) < 0.8:
                recommendations.append(f"Improve {layer.value} validation - current confidence: {validation.get('confidence', 0.0):.2f}")
        
        # Address contradictions
        if contradictions:
            recommendations.append("Resolve identified contradictions between findings")
        
        # Address gaps
        for gap in gaps:
            recommendations.append(f"Address gap: {gap}")
        
        # General recommendations
        recommendations.extend([
            "Conduct additional fact-checking for critical claims",
            "Seek expert review for complex findings",
            "Expand source diversity for better coverage",
            "Perform bias assessment for all sources"
        ])
        
        return recommendations[:10]  # Limit to 10 recommendations
    
    def _calculate_overall_confidence(self, validation_layers: Dict[ValidationLayer, Dict[str, Any]]) -> float:
        """Calculate overall confidence score"""
        if not validation_layers:
            return 0.0
        
        total_confidence = 0.0
        layer_count = 0
        
        for layer, validation in validation_layers.items():
            confidence = validation.get("confidence", 0.0)
            total_confidence += confidence
            layer_count += 1
        
        return total_confidence / layer_count if layer_count > 0 else 0.0
    
    def _calculate_quality_metrics(self, validation_layers: Dict[ValidationLayer, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quality metrics based on validation results"""
        metrics = {}
        
        for metric_name, metric_config in self.quality_metrics.items():
            if metric_name == "accuracy" and ValidationLayer.FACTUAL in validation_layers:
                metrics[metric_name] = validation_layers[ValidationLayer.FACTUAL].get("confidence", 0.0)
            elif metric_name == "reliability" and ValidationLayer.SOURCE in validation_layers:
                metrics[metric_name] = validation_layers[ValidationLayer.SOURCE].get("confidence", 0.0)
            elif metric_name == "completeness" and ValidationLayer.COMPLETENESS in validation_layers:
                metrics[metric_name] = validation_layers[ValidationLayer.COMPLETENESS].get("confidence", 0.0)
            elif metric_name == "consistency" and ValidationLayer.LOGICAL in validation_layers:
                metrics[metric_name] = validation_layers[ValidationLayer.LOGICAL].get("confidence", 0.0)
            elif metric_name == "objectivity" and ValidationLayer.BIAS in validation_layers:
                metrics[metric_name] = validation_layers[ValidationLayer.BIAS].get("confidence", 0.0)
            else:
                metrics[metric_name] = 0.5  # Default value
        
        return metrics
    
    def _generate_quality_assessment(self, validation_result: ValidationReport) -> Dict[str, float]:
        """Generate quality assessment based on validation results"""
        assessment = {}
        
        # Calculate weighted quality score
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, metric_config in self.quality_metrics.items():
            weight = metric_config["weight"]
            score = validation_result.quality_metrics.get(metric_name, 0.0)
            
            total_score += score * weight
            total_weight += weight
        
        overall_quality = total_score / total_weight if total_weight > 0 else 0.0
        
        assessment["overall_quality"] = overall_quality
        assessment["validation_confidence"] = validation_result.overall_confidence
        assessment["contradiction_count"] = len(validation_result.contradictions)
        assessment["gap_count"] = len(validation_result.gaps)
        
        return assessment
    
    async def _generate_improvement_suggestions(self, validation_result: ValidationReport) -> List[str]:
        """Generate improvement suggestions based on validation results"""
        suggestions = []
        
        # Add recommendations from validation report
        suggestions.extend(validation_result.recommendations)
        
        # Generate additional suggestions based on quality metrics
        for metric_name, score in validation_result.quality_metrics.items():
            metric_config = self.quality_metrics.get(metric_name, {})
            threshold = metric_config.get("threshold", 0.7)
            
            if score < threshold:
                suggestions.append(f"Improve {metric_name}: current score {score:.2f} below threshold {threshold}")
        
        # Generate suggestions for contradictions
        if validation_result.contradictions:
            suggestions.append("Investigate and resolve identified contradictions")
        
        # Generate suggestions for gaps
        if validation_result.gaps:
            suggestions.append("Address identified research gaps")
        
        return suggestions[:15]  # Limit to 15 suggestions
    
    def _perform_risk_assessment(self, validation_result: ValidationReport) -> Dict[str, Any]:
        """Perform risk assessment based on validation results"""
        risks = []
        risk_level = "low"
        
        # Assess risks based on validation issues
        if validation_result.overall_confidence < 0.7:
            risks.append("Low validation confidence may indicate unreliable findings")
            risk_level = "medium"
        
        if len(validation_result.contradictions) > 2:
            risks.append("Multiple contradictions suggest potential data quality issues")
            risk_level = "high"
        
        if len(validation_result.gaps) > 3:
            risks.append("Multiple gaps may indicate incomplete research")
            risk_level = "medium"
        
        # Assess quality metric risks
        for metric_name, score in validation_result.quality_metrics.items():
            metric_config = self.quality_metrics.get(metric_name, {})
            threshold = metric_config.get("threshold", 0.7)
            
            if score < threshold:
                risks.append(f"Low {metric_name} score ({score:.2f}) may impact reliability")
        
        return {
            "risk_level": risk_level,
            "risks": risks,
            "mitigation_strategies": self._generate_mitigation_strategies(risks)
        }
    
    def _generate_mitigation_strategies(self, risks: List[str]) -> List[str]:
        """Generate mitigation strategies for identified risks"""
        strategies = []
        
        for risk in risks:
            if "confidence" in risk.lower():
                strategies.append("Conduct additional validation and fact-checking")
            elif "contradiction" in risk.lower():
                strategies.append("Review and resolve contradictory findings")
            elif "gap" in risk.lower():
                strategies.append("Expand research scope to address gaps")
            elif "quality" in risk.lower():
                strategies.append("Improve data quality and source verification")
        
        return strategies
    
    def _extract_validation_result(self, content: str, validation_type: str) -> Dict[str, Any]:
        """Extract validation result from content"""
        # Simplified extraction - in production, use more sophisticated parsing
        
        # Calculate confidence based on content characteristics
        confidence = 0.7  # Base confidence
        
        # Adjust based on content indicators
        if "valid" in content.lower() or "accurate" in content.lower():
            confidence += 0.2
        elif "issue" in content.lower() or "problem" in content.lower():
            confidence -= 0.2
        
        # Extract issues
        issues = []
        if "issue" in content.lower() or "problem" in content.lower():
            issues.append(f"{validation_type} validation identified potential issues")
        
        return {
            "status": "completed",
            "confidence": max(0.0, min(1.0, confidence)),
            "issues": issues,
            "type": validation_type
        }
    
    # Query-specific methods
    async def _fact_check_content(self, query_data: Dict[str, Any]) -> Optional[ValidationOutput]:
        """Fact-check specific content"""
        content = query_data.get("content", "")
        
        if not content:
            return None
        
        validation_result = await self._perform_factual_validation([content], [])
        
        return ValidationOutput(
            agent_type=AgentType.VALIDATION,
            content={"fact_check": validation_result},
            confidence_score=validation_result.get("confidence", 0.7),
            validation_report=ValidationReport(
                layers={ValidationLayer.FACTUAL: validation_result},
                overall_confidence=validation_result.get("confidence", 0.7)
            ),
            quality_assessment={"factual_accuracy": validation_result.get("confidence", 0.7)},
            improvement_suggestions=[],
            risk_assessment={"risk_level": "low", "risks": []}
        )
    
    async def _validate_sources(self, query_data: Dict[str, Any]) -> Optional[ValidationOutput]:
        """Validate specific sources"""
        sources = query_data.get("sources", [])
        
        if not sources:
            return None
        
        validation_result = await self._perform_source_validation(sources)
        
        return ValidationOutput(
            agent_type=AgentType.VALIDATION,
            content={"source_validation": validation_result},
            confidence_score=validation_result.get("confidence", 0.7),
            validation_report=ValidationReport(
                layers={ValidationLayer.SOURCE: validation_result},
                overall_confidence=validation_result.get("confidence", 0.7)
            ),
            quality_assessment={"source_reliability": validation_result.get("confidence", 0.7)},
            improvement_suggestions=[],
            risk_assessment={"risk_level": "low", "risks": []}
        )
    
    async def _analyze_bias(self, query_data: Dict[str, Any]) -> Optional[ValidationOutput]:
        """Analyze bias in content"""
        content = query_data.get("content", "")
        
        if not content:
            return None
        
        validation_result = await self._perform_bias_validation([content], [])
        
        return ValidationOutput(
            agent_type=AgentType.VALIDATION,
            content={"bias_analysis": validation_result},
            confidence_score=validation_result.get("confidence", 0.7),
            validation_report=ValidationReport(
                layers={ValidationLayer.BIAS: validation_result},
                overall_confidence=validation_result.get("confidence", 0.7)
            ),
            quality_assessment={"objectivity": validation_result.get("confidence", 0.7)},
            improvement_suggestions=[],
            risk_assessment={"risk_level": "low", "risks": []}
        ) 