"""
Research Agent for the Multi-Agent Autonomous Research System
"""
import asyncio
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from urllib.parse import urlparse
import re

from base_agent import BaseAgent
from models import (
    AgentType, AgentMessage, ResearchOutput, ResearchSource, 
    ResearchFinding, SourceQuality
)
from config import Config

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """Comprehensive information gatherer and source identifier"""
    
    def __init__(self):
        """Initialize the research agent"""
        super().__init__(AgentType.RESEARCH)
        self.search_strategies = self._load_search_strategies()
        self.source_cache: Dict[str, ResearchSource] = {}
        
    def _load_search_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined search strategies"""
        return {
            "comprehensive": {
                "max_sources": 20,
                "source_types": ["academic", "news", "technical", "expert"],
                "search_depth": "deep",
                "cross_reference": True
            },
            "standard": {
                "max_sources": 10,
                "source_types": ["academic", "news"],
                "search_depth": "medium",
                "cross_reference": False
            },
            "quick": {
                "max_sources": 5,
                "source_types": ["news", "technical"],
                "search_depth": "shallow",
                "cross_reference": False
            }
        }
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Optional[ResearchOutput]:
        """Execute a research task"""
        query = task_data.get("query", "")
        context = task_data.get("context", {})
        workflow_state = task_data.get("workflow_state", {})
        
        if not query:
            logger.error("No query provided for research task")
            return None
        
        self.log_activity("Starting research task", {"query": query})
        
        # Determine search strategy
        strategy = self._determine_search_strategy(context, workflow_state)
        
        # Execute comprehensive research
        research_result = await self._execute_comprehensive_research(query, strategy)
        
        # Analyze coverage and identify gaps
        coverage_analysis = await self._analyze_coverage(query, research_result["sources"])
        
        # Create research findings
        findings = await self._create_research_findings(research_result["sources"], query)
        
        return ResearchOutput(
            agent_type=AgentType.RESEARCH,
            content=research_result,
            confidence_score=research_result.get("confidence_score", 0.8),
            sources=research_result["sources"],
            findings=findings,
            search_strategy=strategy,
            coverage_analysis=coverage_analysis
        )
    
    async def execute_query(self, query_data: Dict[str, Any]) -> Optional[ResearchOutput]:
        """Execute a research query"""
        query_type = query_data.get("type", "search")
        
        if query_type == "search":
            return await self._execute_search_query(query_data)
        elif query_type == "source_validation":
            return await self._validate_source(query_data)
        elif query_type == "coverage_analysis":
            return await self._analyze_coverage_query(query_data)
        else:
            logger.warning(f"Unknown research query type: {query_type}")
            return None
    
    def _determine_search_strategy(
        self, 
        context: Dict[str, Any], 
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine the appropriate search strategy"""
        research_depth = context.get("research_depth", "comprehensive")
        complexity_score = context.get("complexity_score", 0.5)
        
        # Select base strategy
        if research_depth == "comprehensive" or complexity_score > 0.7:
            base_strategy = self.search_strategies["comprehensive"]
        elif research_depth == "standard" or complexity_score > 0.3:
            base_strategy = self.search_strategies["standard"]
        else:
            base_strategy = self.search_strategies["quick"]
        
        # Customize based on domain
        domain = context.get("domain", "general")
        if domain in ["academic", "scientific"]:
            base_strategy["source_types"].extend(["academic", "peer_reviewed"])
        elif domain in ["technology", "engineering"]:
            base_strategy["source_types"].extend(["technical", "patent"])
        
        return base_strategy
    
    async def _execute_comprehensive_research(
        self, 
        query: str, 
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute comprehensive research across multiple sources"""
        self.log_activity("Executing comprehensive research", {"strategy": strategy})
        
        sources = []
        max_sources = strategy.get("max_sources", 10)
        
        # Generate search queries
        search_queries = await self._generate_search_queries(query, strategy)
        
        # Execute searches in parallel
        search_tasks = []
        for search_query in search_queries:
            task = self._search_sources(search_query, strategy)
            search_tasks.append(task)
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process and filter results
        for result in search_results:
            if isinstance(result, list):
                sources.extend(result)
        
        # Remove duplicates and filter by quality
        unique_sources = self._deduplicate_sources(sources)
        filtered_sources = self._filter_sources_by_quality(unique_sources, strategy)
        
        # Limit to max sources
        final_sources = filtered_sources[:max_sources]
        
        # Calculate confidence score
        confidence_score = self._calculate_research_confidence(final_sources, strategy)
        
        return {
            "sources": final_sources,
            "search_queries": search_queries,
            "total_sources_found": len(sources),
            "filtered_sources": len(final_sources),
            "confidence_score": confidence_score,
            "search_strategy": strategy
        }
    
    async def _generate_search_queries(self, query: str, strategy: Dict[str, Any]) -> List[str]:
        """Generate multiple search queries for comprehensive coverage"""
        prompt = f"""
        Generate 5-8 different search queries to comprehensively research this topic:
        
        Original query: {query}
        Search strategy: {strategy}
        
        Create queries that:
        1. Use different keywords and phrases
        2. Target different aspects of the topic
        3. Include both broad and specific searches
        4. Use academic and technical terminology
        5. Include recent developments and trends
        
        Return only the search queries, one per line.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            content = response["content"]
            queries = [line.strip() for line in content.split('\n') if line.strip()]
            return queries[:8]  # Limit to 8 queries
        else:
            # Fallback queries
            return [
                query,
                f"{query} research",
                f"{query} analysis",
                f"{query} latest developments",
                f"{query} expert opinion"
            ]
    
    async def _search_sources(self, search_query: str, strategy: Dict[str, Any]) -> List[ResearchSource]:
        """Search for sources using the given query"""
        sources = []
        
        # Simulate web search (in a real implementation, this would use actual APIs)
        simulated_sources = await self._simulate_web_search(search_query, strategy)
        
        for source_data in simulated_sources:
            source = await self._create_research_source(source_data, search_query)
            if source:
                sources.append(source)
        
        return sources
    
    async def _simulate_web_search(self, query: str, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate web search results (replace with actual API calls)"""
        # This is a simulation - in production, you'd use real search APIs
        # like Google Custom Search, Bing Search, arXiv, PubMed, etc.
        
        prompt = f"""
        Simulate 3-5 high-quality search results for this query:
        
        Query: {query}
        Strategy: {strategy}
        
        Provide realistic search results with:
        - Title
        - URL
        - Brief content summary
        - Author/organization
        - Publication date
        - Quality indicators
        """
        
        output_format = {
            "results": [
                {
                    "title": "string",
                    "url": "string",
                    "content": "string",
                    "author": "string",
                    "publication_date": "string",
                    "quality_indicators": "list of strings"
                }
            ]
        }
        
        response = await self.generate_structured_response_with_gemini(
            prompt=f"Generate search results for: {query}",
            output_format=output_format
        )
        
        if response.get("success") and response.get("structured_data"):
            return response["structured_data"].get("results", [])
        else:
            # Fallback simulation
            return [
                {
                    "title": f"Research on {query}",
                    "url": f"https://example.com/research/{query.replace(' ', '-')}",
                    "content": f"Comprehensive analysis of {query} with detailed findings and implications.",
                    "author": "Research Team",
                    "publication_date": "2024-01-15",
                    "quality_indicators": ["peer_reviewed", "recent", "comprehensive"]
                }
            ]
    
    async def _create_research_source(
        self, 
        source_data: Dict[str, Any], 
        search_query: str
    ) -> Optional[ResearchSource]:
        """Create a ResearchSource object from source data"""
        try:
            # Calculate quality score
            quality_score = self._calculate_source_quality(source_data)
            quality_level = self._determine_quality_level(quality_score)
            
            # Calculate confidence score
            confidence_score = self._calculate_source_confidence(source_data)
            
            # Extract domain
            domain = self._extract_domain(source_data.get("url", ""))
            
            source = ResearchSource(
                url=source_data.get("url"),
                title=source_data.get("title", "Untitled"),
                content=source_data.get("content", ""),
                author=source_data.get("author"),
                publication_date=self._parse_date(source_data.get("publication_date")),
                quality_score=quality_score,
                quality_level=quality_level,
                domain=domain,
                metadata={
                    "search_query": search_query,
                    "quality_indicators": source_data.get("quality_indicators", []),
                    "extracted_at": datetime.now().isoformat()
                },
                confidence_score=confidence_score
            )
            
            return source
            
        except Exception as e:
            logger.error(f"Error creating research source: {str(e)}")
            return None
    
    def _calculate_source_quality(self, source_data: Dict[str, Any]) -> float:
        """Calculate quality score for a source"""
        score = 0.5  # Base score
        
        # Quality indicators
        indicators = source_data.get("quality_indicators", [])
        for indicator in indicators:
            if indicator in ["peer_reviewed", "academic", "expert"]:
                score += 0.2
            elif indicator in ["recent", "comprehensive", "detailed"]:
                score += 0.1
            elif indicator in ["verified", "authoritative"]:
                score += 0.15
        
        # URL quality
        url = source_data.get("url", "")
        if url:
            if any(domain in url.lower() for domain in [".edu", ".gov", ".org"]):
                score += 0.1
            elif any(domain in url.lower() for domain in [".com", ".net"]):
                score += 0.05
        
        # Content length
        content = source_data.get("content", "")
        if len(content) > 500:
            score += 0.1
        elif len(content) < 100:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _determine_quality_level(self, quality_score: float) -> SourceQuality:
        """Determine quality level based on score"""
        if quality_score >= 0.8:
            return SourceQuality.ACADEMIC
        elif quality_score >= 0.6:
            return SourceQuality.HIGH
        elif quality_score >= 0.4:
            return SourceQuality.MEDIUM
        else:
            return SourceQuality.LOW
    
    def _calculate_source_confidence(self, source_data: Dict[str, Any]) -> float:
        """Calculate confidence score for source reliability"""
        confidence = 0.7  # Base confidence
        
        # Author credibility
        author = source_data.get("author", "")
        if author and len(author) > 0:
            confidence += 0.1
        
        # Publication date
        if source_data.get("publication_date"):
            confidence += 0.1
        
        # Content completeness
        content = source_data.get("content", "")
        if len(content) > 200:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except:
            return "unknown"
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        try:
            # Try common date formats
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%B %d, %Y"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None
        except:
            return None
    
    def _deduplicate_sources(self, sources: List[ResearchSource]) -> List[ResearchSource]:
        """Remove duplicate sources based on URL and title similarity"""
        unique_sources = []
        seen_urls = set()
        seen_titles = set()
        
        for source in sources:
            # Check URL
            if source.url and source.url in seen_urls:
                continue
            
            # Check title similarity
            title_lower = source.title.lower()
            if any(self._calculate_similarity(title_lower, seen_title) > 0.8 
                   for seen_title in seen_titles):
                continue
            
            unique_sources.append(source)
            if source.url:
                seen_urls.add(source.url)
            seen_titles.add(title_lower)
        
        return unique_sources
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple similarity calculation
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _filter_sources_by_quality(
        self, 
        sources: List[ResearchSource], 
        strategy: Dict[str, Any]
    ) -> List[ResearchSource]:
        """Filter sources based on quality requirements"""
        min_quality = strategy.get("min_source_quality", 0.6)
        
        filtered = [source for source in sources if source.quality_score >= min_quality]
        
        # Sort by quality score (highest first)
        filtered.sort(key=lambda x: x.quality_score, reverse=True)
        
        return filtered
    
    async def _create_research_findings(
        self, 
        sources: List[ResearchSource], 
        query: str
    ) -> List[ResearchFinding]:
        """Create research findings from sources"""
        findings = []
        
        # Group sources by topic/category
        source_groups = self._group_sources_by_topic(sources, query)
        
        for group_name, group_sources in source_groups.items():
            finding = await self._create_finding_from_group(group_name, group_sources, query)
            if finding:
                findings.append(finding)
        
        return findings
    
    def _group_sources_by_topic(
        self, 
        sources: List[ResearchSource], 
        query: str
    ) -> Dict[str, List[ResearchSource]]:
        """Group sources by topic/category"""
        groups = {}
        
        for source in sources:
            # Simple grouping based on domain and content keywords
            group_key = self._determine_source_group(source, query)
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(source)
        
        return groups
    
    def _determine_source_group(self, source: ResearchSource, query: str) -> str:
        """Determine which group a source belongs to"""
        # Extract key terms from query
        query_terms = set(query.lower().split())
        
        # Check content for query terms
        content_lower = source.content.lower()
        matching_terms = query_terms.intersection(set(content_lower.split()))
        
        if len(matching_terms) >= len(query_terms) * 0.5:
            return "primary"
        elif len(matching_terms) >= len(query_terms) * 0.2:
            return "secondary"
        else:
            return "related"
    
    async def _create_finding_from_group(
        self, 
        group_name: str, 
        sources: List[ResearchSource], 
        query: str
    ) -> Optional[ResearchFinding]:
        """Create a research finding from a group of sources"""
        if not sources:
            return None
        
        # Synthesize content from sources
        prompt = f"""
        Synthesize the following sources into a coherent research finding:
        
        Query: {query}
        Group: {group_name}
        Sources: {[s.title for s in sources]}
        
        Create a finding that:
        1. Summarizes the key information
        2. Identifies patterns and insights
        3. Highlights important details
        4. Maintains source attribution
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(sources, query)
            
            # Calculate confidence score
            confidence_score = sum(s.confidence_score for s in sources) / len(sources)
            
            finding = ResearchFinding(
                title=f"{group_name.title()} Finding: {query}",
                content=response["content"],
                sources=sources,
                confidence_score=confidence_score,
                relevance_score=relevance_score,
                category=group_name,
                tags=[query, group_name]
            )
            
            return finding
        
        return None
    
    def _calculate_relevance_score(self, sources: List[ResearchSource], query: str) -> float:
        """Calculate relevance score for a group of sources"""
        if not sources:
            return 0.0
        
        # Average quality and confidence scores
        avg_quality = sum(s.quality_score for s in sources) / len(sources)
        avg_confidence = sum(s.confidence_score for s in sources) / len(sources)
        
        # Query relevance (simplified)
        query_terms = set(query.lower().split())
        total_relevance = 0
        
        for source in sources:
            content_terms = set(source.content.lower().split())
            matching = len(query_terms.intersection(content_terms))
            relevance = matching / len(query_terms) if query_terms else 0
            total_relevance += relevance
        
        avg_relevance = total_relevance / len(sources)
        
        # Combine scores
        final_score = (avg_quality * 0.4) + (avg_confidence * 0.3) + (avg_relevance * 0.3)
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_research_confidence(
        self, 
        sources: List[ResearchSource], 
        strategy: Dict[str, Any]
    ) -> float:
        """Calculate overall research confidence score"""
        if not sources:
            return 0.0
        
        # Source quality
        avg_quality = sum(s.quality_score for s in sources) / len(sources)
        
        # Source diversity
        domains = set(s.domain for s in sources if s.domain)
        diversity_score = min(len(domains) / 5, 1.0)  # Normalize to 0-1
        
        # Coverage completeness
        expected_sources = strategy.get("max_sources", 10)
        coverage_score = min(len(sources) / expected_sources, 1.0)
        
        # Confidence scores
        avg_confidence = sum(s.confidence_score for s in sources) / len(sources)
        
        # Weighted combination
        final_score = (
            avg_quality * 0.3 +
            diversity_score * 0.2 +
            coverage_score * 0.2 +
            avg_confidence * 0.3
        )
        
        return max(0.0, min(1.0, final_score))
    
    async def _analyze_coverage(self, query: str, sources: List[ResearchSource]) -> Dict[str, Any]:
        """Analyze research coverage and identify gaps"""
        prompt = f"""
        Analyze the research coverage for this query:
        
        Query: {query}
        Sources found: {len(sources)}
        Source domains: {[s.domain for s in sources]}
        
        Identify:
        1. Coverage gaps
        2. Missing perspectives
        3. Areas needing more research
        4. Source diversity assessment
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        return {
            "coverage_analysis": response.get("content", ""),
            "total_sources": len(sources),
            "domain_diversity": len(set(s.domain for s in sources if s.domain)),
            "quality_distribution": self._analyze_quality_distribution(sources),
            "gaps_identified": True  # Would be extracted from response
        }
    
    def _analyze_quality_distribution(self, sources: List[ResearchSource]) -> Dict[str, int]:
        """Analyze the distribution of source quality levels"""
        distribution = {
            "academic": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for source in sources:
            distribution[source.quality_level.value] += 1
        
        return distribution
    
    async def _execute_search_query(self, query_data: Dict[str, Any]) -> Optional[ResearchOutput]:
        """Execute a specific search query"""
        query = query_data.get("query", "")
        if not query:
            return None
        
        strategy = self.search_strategies["standard"]
        result = await self._execute_comprehensive_research(query, strategy)
        
        return ResearchOutput(
            agent_type=AgentType.RESEARCH,
            content=result,
            confidence_score=result.get("confidence_score", 0.7),
            sources=result["sources"],
            findings=[],
            search_strategy=strategy,
            coverage_analysis={}
        )
    
    async def _validate_source(self, query_data: Dict[str, Any]) -> Optional[ResearchOutput]:
        """Validate a specific source"""
        source_url = query_data.get("url", "")
        if not source_url:
            return None
        
        # Simulate source validation
        validation_result = {
            "url": source_url,
            "is_valid": True,
            "quality_score": 0.8,
            "validation_details": "Source appears to be valid and reliable"
        }
        
        return ResearchOutput(
            agent_type=AgentType.RESEARCH,
            content=validation_result,
            confidence_score=0.9,
            sources=[],
            findings=[],
            search_strategy={},
            coverage_analysis={}
        )
    
    async def _analyze_coverage_query(self, query_data: Dict[str, Any]) -> Optional[ResearchOutput]:
        """Analyze coverage for a specific query"""
        query = query_data.get("query", "")
        sources = query_data.get("sources", [])
        
        coverage_analysis = await self._analyze_coverage(query, sources)
        
        return ResearchOutput(
            agent_type=AgentType.RESEARCH,
            content=coverage_analysis,
            confidence_score=0.8,
            sources=sources,
            findings=[],
            search_strategy={},
            coverage_analysis=coverage_analysis
        ) 