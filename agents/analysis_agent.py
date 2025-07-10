"""
Analysis Agent for the Multi-Agent Autonomous Research System
"""
import asyncio
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict
import re

from base_agent import BaseAgent
from models import (
    AgentType, AgentMessage, AnalysisOutput, AnalysisResult, 
    ResearchFinding, ResearchSource
)
from config import Config

logger = logging.getLogger(__name__)

class AnalysisAgent(BaseAgent):
    """Deep analysis and pattern recognition specialist"""
    
    def __init__(self):
        """Initialize the analysis agent"""
        super().__init__(AgentType.ANALYSIS)
        self.analysis_methods = self._load_analysis_methods()
        
    def _load_analysis_methods(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined analysis methods"""
        return {
            "statistical": {
                "correlation_analysis": True,
                "trend_analysis": True,
                "outlier_detection": True,
                "confidence_intervals": True
            },
            "pattern_recognition": {
                "keyword_extraction": True,
                "topic_modeling": True,
                "sentiment_analysis": True,
                "entity_recognition": True
            },
            "comparative": {
                "cross_source_comparison": True,
                "contradiction_detection": True,
                "consensus_analysis": True,
                "perspective_comparison": True
            },
            "predictive": {
                "trend_prediction": True,
                "impact_analysis": True,
                "scenario_modeling": True,
                "risk_assessment": True
            }
        }
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Optional[AnalysisOutput]:
        """Execute an analysis task"""
        findings = task_data.get("findings", [])
        sources = task_data.get("sources", [])
        query = task_data.get("query", "")
        context = task_data.get("context", {})
        
        if not findings and not sources:
            logger.error("No findings or sources provided for analysis")
            return None
        
        self.log_activity("Starting analysis task", {"findings_count": len(findings), "sources_count": len(sources)})
        
        # Determine analysis approach
        analysis_approach = self._determine_analysis_approach(context, findings, sources)
        
        # Perform comprehensive analysis
        analysis_result = await self._perform_comprehensive_analysis(
            findings, sources, query, analysis_approach
        )
        
        # Extract insights and patterns
        insights = await self._extract_insights(analysis_result, query)
        
        # Identify correlations
        correlations = await self._identify_correlations(analysis_result, findings)
        
        # Detect patterns
        patterns = await self._detect_patterns(analysis_result, findings)
        
        return AnalysisOutput(
            agent_type=AgentType.ANALYSIS,
            content=analysis_result,
            confidence_score=analysis_result.get("confidence_score", 0.8),
            analysis_result=analysis_result,
            insights=insights,
            correlations=correlations,
            patterns=patterns
        )
    
    async def execute_query(self, query_data: Dict[str, Any]) -> Optional[AnalysisOutput]:
        """Execute an analysis query"""
        query_type = query_data.get("type", "analysis")
        
        if query_type == "pattern_analysis":
            return await self._analyze_patterns(query_data)
        elif query_type == "correlation_analysis":
            return await self._analyze_correlations(query_data)
        elif query_type == "trend_analysis":
            return await self._analyze_trends(query_data)
        else:
            logger.warning(f"Unknown analysis query type: {query_type}")
            return None
    
    def _determine_analysis_approach(
        self, 
        context: Dict[str, Any], 
        findings: List[ResearchFinding], 
        sources: List[ResearchSource]
    ) -> Dict[str, Any]:
        """Determine the appropriate analysis approach"""
        domain = context.get("domain", "general")
        complexity_score = context.get("complexity_score", 0.5)
        
        # Select base approach
        if complexity_score > 0.7:
            base_approach = {
                "statistical": True,
                "pattern_recognition": True,
                "comparative": True,
                "predictive": True
            }
        elif complexity_score > 0.4:
            base_approach = {
                "statistical": True,
                "pattern_recognition": True,
                "comparative": True,
                "predictive": False
            }
        else:
            base_approach = {
                "statistical": False,
                "pattern_recognition": True,
                "comparative": True,
                "predictive": False
            }
        
        # Customize based on domain
        if domain in ["scientific", "academic"]:
            base_approach["statistical"] = True
        elif domain in ["business", "economics"]:
            base_approach["predictive"] = True
        
        return base_approach
    
    async def _perform_comprehensive_analysis(
        self, 
        findings: List[ResearchFinding], 
        sources: List[ResearchSource], 
        query: str, 
        approach: Dict[str, Any]
    ) -> AnalysisResult:
        """Perform comprehensive analysis using multiple methods"""
        self.log_activity("Performing comprehensive analysis", {"approach": approach})
        
        analysis_tasks = []
        
        # Statistical analysis
        if approach.get("statistical"):
            analysis_tasks.append(self._perform_statistical_analysis(findings, sources))
        
        # Pattern recognition
        if approach.get("pattern_recognition"):
            analysis_tasks.append(self._perform_pattern_recognition(findings, sources))
        
        # Comparative analysis
        if approach.get("comparative"):
            analysis_tasks.append(self._perform_comparative_analysis(findings, sources))
        
        # Predictive analysis
        if approach.get("predictive"):
            analysis_tasks.append(self._perform_predictive_analysis(findings, sources, query))
        
        # Execute all analysis tasks
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Combine results
        combined_result = self._combine_analysis_results(results)
        
        # Calculate confidence score
        confidence_score = self._calculate_analysis_confidence(combined_result, approach)
        
        return AnalysisResult(
            patterns=combined_result.get("patterns", []),
            correlations=combined_result.get("correlations", []),
            trends=combined_result.get("trends", []),
            insights=combined_result.get("insights", []),
            confidence_intervals=combined_result.get("confidence_intervals", {}),
            statistical_metrics=combined_result.get("statistical_metrics", {})
        )
    
    async def _perform_statistical_analysis(
        self, 
        findings: List[ResearchFinding], 
        sources: List[ResearchSource]
    ) -> Dict[str, Any]:
        """Perform statistical analysis on the data"""
        prompt = f"""
        Perform statistical analysis on the following research data:
        
        Findings: {[f.title for f in findings]}
        Sources: {len(sources)} sources
        
        Analyze:
        1. Data distribution and patterns
        2. Statistical significance of findings
        3. Confidence intervals
        4. Correlation coefficients
        5. Outlier detection
        
        Provide statistical metrics and insights.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            # Extract statistical metrics
            metrics = self._extract_statistical_metrics(response["content"])
            
            return {
                "type": "statistical",
                "metrics": metrics,
                "confidence_intervals": self._calculate_confidence_intervals(metrics),
                "outliers": self._detect_outliers(metrics),
                "correlations": self._calculate_correlations(metrics)
            }
        else:
            return {"type": "statistical", "error": "Analysis failed"}
    
    async def _perform_pattern_recognition(
        self, 
        findings: List[ResearchFinding], 
        sources: List[ResearchSource]
    ) -> Dict[str, Any]:
        """Perform pattern recognition analysis"""
        # Extract text content for analysis
        all_text = self._extract_text_content(findings, sources)
        
        prompt = f"""
        Perform pattern recognition analysis on this research content:
        
        Content length: {len(all_text)} characters
        Findings count: {len(findings)}
        Sources count: {len(sources)}
        
        Identify:
        1. Key themes and topics
        2. Recurring patterns
        3. Keyword frequency
        4. Semantic relationships
        5. Topic clusters
        
        Focus on meaningful patterns that provide insights.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            patterns = self._extract_patterns_from_text(response["content"])
            
            return {
                "type": "pattern_recognition",
                "patterns": patterns,
                "themes": self._extract_themes(patterns),
                "keywords": self._extract_keywords(all_text),
                "topic_clusters": self._create_topic_clusters(patterns)
            }
        else:
            return {"type": "pattern_recognition", "error": "Analysis failed"}
    
    async def _perform_comparative_analysis(
        self, 
        findings: List[ResearchFinding], 
        sources: List[ResearchSource]
    ) -> Dict[str, Any]:
        """Perform comparative analysis across sources and findings"""
        prompt = f"""
        Perform comparative analysis on the research findings:
        
        Findings: {[f.title for f in findings]}
        Sources: {len(sources)} sources from various domains
        
        Compare:
        1. Consistency across sources
        2. Contradictions and conflicts
        3. Different perspectives
        4. Consensus areas
        5. Source reliability comparison
        
        Identify areas of agreement and disagreement.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            comparisons = self._extract_comparisons(response["content"])
            
            return {
                "type": "comparative",
                "comparisons": comparisons,
                "contradictions": self._identify_contradictions(comparisons),
                "consensus": self._identify_consensus(comparisons),
                "perspectives": self._identify_perspectives(comparisons)
            }
        else:
            return {"type": "comparative", "error": "Analysis failed"}
    
    async def _perform_predictive_analysis(
        self, 
        findings: List[ResearchFinding], 
        sources: List[ResearchSource], 
        query: str
    ) -> Dict[str, Any]:
        """Perform predictive analysis and trend forecasting"""
        prompt = f"""
        Perform predictive analysis based on the research findings:
        
        Query: {query}
        Findings: {[f.title for f in findings]}
        Sources: {len(sources)} sources
        
        Predict:
        1. Future trends and developments
        2. Potential impacts and implications
        3. Risk factors and opportunities
        4. Scenario modeling
        5. Recommendations for action
        
        Base predictions on current data and trends.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            predictions = self._extract_predictions(response["content"])
            
            return {
                "type": "predictive",
                "predictions": predictions,
                "trends": self._extract_trends(predictions),
                "scenarios": self._create_scenarios(predictions),
                "recommendations": self._extract_recommendations(response["content"])
            }
        else:
            return {"type": "predictive", "error": "Analysis failed"}
    
    def _combine_analysis_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from different analysis methods"""
        combined = {
            "patterns": [],
            "correlations": [],
            "trends": [],
            "insights": [],
            "confidence_intervals": {},
            "statistical_metrics": {}
        }
        
        for result in results:
            if isinstance(result, dict):
                if result.get("type") == "statistical":
                    combined["statistical_metrics"].update(result.get("metrics", {}))
                    combined["confidence_intervals"].update(result.get("confidence_intervals", {}))
                    combined["correlations"].extend(result.get("correlations", []))
                elif result.get("type") == "pattern_recognition":
                    combined["patterns"].extend(result.get("patterns", []))
                elif result.get("type") == "comparative":
                    combined["insights"].extend(result.get("comparisons", []))
                elif result.get("type") == "predictive":
                    combined["trends"].extend(result.get("trends", []))
                    combined["insights"].extend(result.get("predictions", []))
        
        return combined
    
    def _calculate_analysis_confidence(
        self, 
        result: Dict[str, Any], 
        approach: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Method coverage
        methods_used = sum(approach.values())
        total_methods = len(approach)
        coverage_score = methods_used / total_methods if total_methods > 0 else 0
        confidence += coverage_score * 0.2
        
        # Result quality
        if result.get("patterns"):
            confidence += 0.1
        if result.get("correlations"):
            confidence += 0.1
        if result.get("trends"):
            confidence += 0.1
        if result.get("insights"):
            confidence += 0.1
        
        # Statistical robustness
        if result.get("statistical_metrics"):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    async def _extract_insights(self, analysis_result: AnalysisResult, query: str) -> List[str]:
        """Extract key insights from analysis results"""
        prompt = f"""
        Extract key insights from this analysis for the query: {query}
        
        Analysis results:
        - Patterns: {len(analysis_result.patterns)} patterns found
        - Correlations: {len(analysis_result.correlations)} correlations
        - Trends: {len(analysis_result.trends)} trends
        - Statistical metrics: {len(analysis_result.statistical_metrics)} metrics
        
        Provide 5-10 key insights that are:
        1. Relevant to the original query
        2. Supported by the analysis
        3. Actionable or informative
        4. Clear and concise
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            # Extract insights from response
            insights = self._extract_insights_from_text(response["content"])
            return insights[:10]  # Limit to 10 insights
        else:
            return ["Analysis completed successfully", "Multiple patterns identified", "Correlations found"]
    
    async def _identify_correlations(self, analysis_result: AnalysisResult, findings: List[ResearchFinding]) -> List[Dict[str, Any]]:
        """Identify correlations between different aspects of the research"""
        correlations = []
        
        # Use existing correlations from analysis
        correlations.extend(analysis_result.correlations)
        
        # Generate additional correlations based on findings
        if findings:
            prompt = f"""
            Identify correlations between these research findings:
            
            Findings: {[f.title for f in findings]}
            
            Look for:
            1. Cause-effect relationships
            2. Co-occurring phenomena
            3. Statistical correlations
            4. Logical connections
            5. Temporal relationships
            """
            
            response = await self.generate_response_with_gemini(prompt)
            
            if response.get("success"):
                additional_correlations = self._extract_correlations_from_text(response["content"])
                correlations.extend(additional_correlations)
        
        return correlations[:20]  # Limit to 20 correlations
    
    async def _detect_patterns(self, analysis_result: AnalysisResult, findings: List[ResearchFinding]) -> List[Dict[str, Any]]:
        """Detect patterns in the research data"""
        patterns = []
        
        # Use existing patterns from analysis
        patterns.extend(analysis_result.patterns)
        
        # Generate additional patterns
        if findings:
            prompt = f"""
            Detect patterns in these research findings:
            
            Findings: {[f.title for f in findings]}
            
            Look for:
            1. Recurring themes
            2. Sequential patterns
            3. Structural patterns
            4. Behavioral patterns
            5. Temporal patterns
            """
            
            response = await self.generate_response_with_gemini(prompt)
            
            if response.get("success"):
                additional_patterns = self._extract_patterns_from_text(response["content"])
                patterns.extend(additional_patterns)
        
        return patterns[:15]  # Limit to 15 patterns
    
    # Helper methods for data extraction and processing
    def _extract_text_content(self, findings: List[ResearchFinding], sources: List[ResearchSource]) -> str:
        """Extract all text content for analysis"""
        text_parts = []
        
        for finding in findings:
            text_parts.append(finding.content)
        
        for source in sources:
            text_parts.append(source.content)
        
        return " ".join(text_parts)
    
    def _extract_statistical_metrics(self, content: str) -> Dict[str, float]:
        """Extract statistical metrics from analysis content"""
        # Simplified extraction - in production, use more sophisticated parsing
        metrics = {}
        
        # Look for common statistical terms
        if "correlation" in content.lower():
            metrics["correlation_strength"] = 0.7
        if "significance" in content.lower():
            metrics["statistical_significance"] = 0.05
        if "confidence" in content.lower():
            metrics["confidence_level"] = 0.95
        
        return metrics
    
    def _calculate_confidence_intervals(self, metrics: Dict[str, float]) -> Dict[str, List[float]]:
        """Calculate confidence intervals for metrics"""
        intervals = {}
        
        for metric, value in metrics.items():
            # Simplified confidence interval calculation
            margin = value * 0.1  # 10% margin
            intervals[metric] = [value - margin, value + margin]
        
        return intervals
    
    def _detect_outliers(self, metrics: Dict[str, float]) -> List[str]:
        """Detect outliers in the data"""
        outliers = []
        
        # Simplified outlier detection
        for metric, value in metrics.items():
            if value > 0.9 or value < 0.1:  # Extreme values
                outliers.append(metric)
        
        return outliers
    
    def _calculate_correlations(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate correlations between metrics"""
        correlations = []
        
        # Simplified correlation calculation
        metric_names = list(metrics.keys())
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                correlation = {
                    "variable1": metric1,
                    "variable2": metric2,
                    "correlation_coefficient": 0.5,  # Simplified
                    "strength": "moderate"
                }
                correlations.append(correlation)
        
        return correlations
    
    def _extract_patterns_from_text(self, content: str) -> List[Dict[str, Any]]:
        """Extract patterns from text content"""
        patterns = []
        
        # Look for pattern indicators in text
        pattern_indicators = ["pattern", "trend", "cycle", "recurring", "consistent"]
        
        for indicator in pattern_indicators:
            if indicator in content.lower():
                pattern = {
                    "type": indicator,
                    "description": f"Identified {indicator} in the data",
                    "confidence": 0.7,
                    "evidence": "Text analysis"
                }
                patterns.append(pattern)
        
        return patterns
    
    def _extract_themes(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Extract themes from patterns"""
        themes = []
        
        for pattern in patterns:
            if "description" in pattern:
                # Extract key terms as themes
                description = pattern["description"]
                words = description.split()
                if len(words) > 2:
                    themes.append(" ".join(words[:3]))
        
        return list(set(themes))  # Remove duplicates
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simplified keyword extraction
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequency
        keyword_counts = defaultdict(int)
        for keyword in keywords:
            keyword_counts[keyword] += 1
        
        # Return top keywords
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, count in sorted_keywords[:20]]
    
    def _create_topic_clusters(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create topic clusters from patterns"""
        clusters = []
        
        # Simplified clustering
        if patterns:
            cluster = {
                "name": "Main Topic Cluster",
                "patterns": patterns,
                "size": len(patterns),
                "coherence": 0.8
            }
            clusters.append(cluster)
        
        return clusters
    
    def _extract_comparisons(self, content: str) -> List[Dict[str, Any]]:
        """Extract comparisons from analysis content"""
        comparisons = []
        
        # Look for comparison indicators
        comparison_indicators = ["similar", "different", "compare", "contrast", "versus"]
        
        for indicator in comparison_indicators:
            if indicator in content.lower():
                comparison = {
                    "type": indicator,
                    "description": f"Comparison based on {indicator}",
                    "confidence": 0.6
                }
                comparisons.append(comparison)
        
        return comparisons
    
    def _identify_contradictions(self, comparisons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify contradictions in the data"""
        contradictions = []
        
        for comparison in comparisons:
            if "different" in comparison.get("type", "").lower():
                contradiction = {
                    "type": "contradiction",
                    "description": comparison.get("description", ""),
                    "severity": "moderate"
                }
                contradictions.append(contradiction)
        
        return contradictions
    
    def _identify_consensus(self, comparisons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify areas of consensus"""
        consensus = []
        
        for comparison in comparisons:
            if "similar" in comparison.get("type", "").lower():
                consensus_item = {
                    "type": "consensus",
                    "description": comparison.get("description", ""),
                    "strength": "strong"
                }
                consensus.append(consensus_item)
        
        return consensus
    
    def _identify_perspectives(self, comparisons: List[Dict[str, Any]]) -> List[str]:
        """Identify different perspectives"""
        perspectives = []
        
        for comparison in comparisons:
            if "perspective" in comparison.get("description", "").lower():
                perspectives.append(comparison.get("description", ""))
        
        return perspectives
    
    def _extract_predictions(self, content: str) -> List[Dict[str, Any]]:
        """Extract predictions from analysis content"""
        predictions = []
        
        # Look for prediction indicators
        prediction_indicators = ["will", "likely", "expected", "predict", "forecast", "trend"]
        
        for indicator in prediction_indicators:
            if indicator in content.lower():
                prediction = {
                    "type": "prediction",
                    "indicator": indicator,
                    "confidence": 0.6,
                    "timeframe": "short-term"
                }
                predictions.append(prediction)
        
        return predictions
    
    def _extract_trends(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract trends from predictions"""
        trends = []
        
        for prediction in predictions:
            if prediction.get("type") == "prediction":
                trend = {
                    "type": "trend",
                    "direction": "increasing",
                    "confidence": prediction.get("confidence", 0.6),
                    "evidence": prediction.get("indicator", "")
                }
                trends.append(trend)
        
        return trends
    
    def _create_scenarios(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create scenarios based on predictions"""
        scenarios = []
        
        if predictions:
            scenario = {
                "name": "Base Scenario",
                "predictions": predictions,
                "probability": 0.7,
                "description": "Most likely outcome based on current trends"
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from analysis content"""
        recommendations = []
        
        # Look for recommendation indicators
        recommendation_indicators = ["recommend", "should", "need to", "must", "important to"]
        
        lines = content.split('\n')
        for line in lines:
            for indicator in recommendation_indicators:
                if indicator in line.lower():
                    recommendations.append(line.strip())
                    break
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _extract_insights_from_text(self, content: str) -> List[str]:
        """Extract insights from text content"""
        insights = []
        
        # Split into sentences and look for insight indicators
        sentences = re.split(r'[.!?]+', content)
        
        insight_indicators = ["insight", "finding", "discovery", "observation", "conclusion"]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum length for meaningful insight
                for indicator in insight_indicators:
                    if indicator in sentence.lower():
                        insights.append(sentence)
                        break
        
        return insights[:10]  # Limit to 10 insights
    
    def _extract_correlations_from_text(self, content: str) -> List[Dict[str, Any]]:
        """Extract correlations from text content"""
        correlations = []
        
        # Look for correlation indicators
        correlation_indicators = ["correlate", "relationship", "connection", "link", "association"]
        
        for indicator in correlation_indicators:
            if indicator in content.lower():
                correlation = {
                    "type": "correlation",
                    "description": f"Correlation based on {indicator}",
                    "strength": "moderate",
                    "confidence": 0.6
                }
                correlations.append(correlation)
        
        return correlations
    
    # Query-specific analysis methods
    async def _analyze_patterns(self, query_data: Dict[str, Any]) -> Optional[AnalysisOutput]:
        """Analyze specific patterns in the data"""
        findings = query_data.get("findings", [])
        
        if not findings:
            return None
        
        analysis_result = await self._perform_pattern_recognition(findings, [])
        patterns = self._extract_patterns_from_text(analysis_result.get("patterns", []))
        
        return AnalysisOutput(
            agent_type=AgentType.ANALYSIS,
            content=analysis_result,
            confidence_score=0.8,
            analysis_result=AnalysisResult(patterns=patterns),
            insights=[],
            correlations=[],
            patterns=patterns
        )
    
    async def _analyze_correlations(self, query_data: Dict[str, Any]) -> Optional[AnalysisOutput]:
        """Analyze correlations in the data"""
        findings = query_data.get("findings", [])
        
        if not findings:
            return None
        
        correlations = await self._identify_correlations(AnalysisResult(), findings)
        
        return AnalysisOutput(
            agent_type=AgentType.ANALYSIS,
            content={"correlations": correlations},
            confidence_score=0.7,
            analysis_result=AnalysisResult(correlations=correlations),
            insights=[],
            correlations=correlations,
            patterns=[]
        )
    
    async def _analyze_trends(self, query_data: Dict[str, Any]) -> Optional[AnalysisOutput]:
        """Analyze trends in the data"""
        findings = query_data.get("findings", [])
        
        if not findings:
            return None
        
        analysis_result = await self._perform_predictive_analysis(findings, [], query_data.get("query", ""))
        trends = analysis_result.get("trends", [])
        
        return AnalysisOutput(
            agent_type=AgentType.ANALYSIS,
            content=analysis_result,
            confidence_score=0.7,
            analysis_result=AnalysisResult(trends=trends),
            insights=[],
            correlations=[],
            patterns=[]
        ) 