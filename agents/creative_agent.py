"""
Creative Agent for the Multi-Agent Autonomous Research System
"""
import asyncio
from typing import Dict, List, Any, Optional
import logging
import random

from base_agent import BaseAgent
from models import (
    AgentType, AgentMessage, CreativeOutput, CreativeConcept
)
from config import Config

logger = logging.getLogger(__name__)

class CreativeAgent(BaseAgent):
    """Innovation and creative problem-solving specialist"""
    
    def __init__(self):
        """Initialize the creative agent"""
        super().__init__(AgentType.CREATIVE)
        self.creativity_methods = self._load_creativity_methods()
        self.innovation_frameworks = self._load_innovation_frameworks()
        
    def _load_creativity_methods(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined creativity methods"""
        return {
            "scamper": {
                "substitute": "What can be substituted?",
                "combine": "What can be combined?",
                "adapt": "What can be adapted?",
                "modify": "What can be modified?",
                "put_to_other_uses": "What other uses can this have?",
                "eliminate": "What can be eliminated?",
                "reverse": "What can be reversed or rearranged?"
            },
            "biomimicry": {
                "nature_inspiration": "How does nature solve this?",
                "biological_analogies": "What biological systems are similar?",
                "evolutionary_principles": "How would evolution approach this?"
            },
            "lateral_thinking": {
                "random_stimulation": "Use random words to stimulate ideas",
                "provocation": "Make provocative statements",
                "challenge_assumptions": "Question basic assumptions"
            },
            "design_thinking": {
                "empathize": "Understand user needs",
                "define": "Define the problem clearly",
                "ideate": "Generate many ideas",
                "prototype": "Create quick prototypes",
                "test": "Test and iterate"
            }
        }
    
    def _load_innovation_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Load innovation frameworks"""
        return {
            "disruptive_innovation": {
                "description": "Create solutions that disrupt existing markets",
                "principles": ["simplicity", "accessibility", "affordability", "convenience"]
            },
            "incremental_innovation": {
                "description": "Improve existing solutions step by step",
                "principles": ["optimization", "refinement", "enhancement", "efficiency"]
            },
            "radical_innovation": {
                "description": "Create completely new approaches",
                "principles": ["paradigm_shift", "novel_technology", "new_market", "breakthrough"]
            },
            "open_innovation": {
                "description": "Collaborate across boundaries",
                "principles": ["collaboration", "diversity", "openness", "ecosystem"]
            }
        }
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Optional[CreativeOutput]:
        """Execute a creative task"""
        query = task_data.get("query", "")
        findings = task_data.get("findings", [])
        context = task_data.get("context", {})
        
        if not query:
            logger.error("No query provided for creative task")
            return None
        
        self.log_activity("Starting creative task", {"query": query})
        
        # Determine creativity approach
        creativity_approach = self._determine_creativity_approach(context, findings)
        
        # Generate creative concepts
        concepts = await self._generate_creative_concepts(query, findings, creativity_approach)
        
        # Calculate innovation metrics
        innovation_metrics = self._calculate_innovation_metrics(concepts)
        
        # Determine synthesis approach
        synthesis_approach = self._determine_synthesis_approach(concepts, creativity_approach)
        
        # Generate cross-domain insights
        cross_domain_insights = await self._generate_cross_domain_insights(concepts, query)
        
        return CreativeOutput(
            agent_type=AgentType.CREATIVE,
            content={
                "concepts": [c.dict() for c in concepts],
                "innovation_metrics": innovation_metrics,
                "synthesis_approach": synthesis_approach
            },
            confidence_score=self._calculate_creative_confidence(concepts, innovation_metrics),
            concepts=concepts,
            innovation_metrics=innovation_metrics,
            synthesis_approach=synthesis_approach,
            cross_domain_insights=cross_domain_insights
        )
    
    async def execute_query(self, query_data: Dict[str, Any]) -> Optional[CreativeOutput]:
        """Execute a creative query"""
        query_type = query_data.get("type", "ideation")
        
        if query_type == "ideation":
            return await self._generate_ideas(query_data)
        elif query_type == "problem_solving":
            return await self._solve_problem_creatively(query_data)
        elif query_type == "innovation_analysis":
            return await self._analyze_innovation_potential(query_data)
        else:
            logger.warning(f"Unknown creative query type: {query_type}")
            return None
    
    def _determine_creativity_approach(
        self, 
        context: Dict[str, Any], 
        findings: List[Any]
    ) -> Dict[str, Any]:
        """Determine the appropriate creativity approach"""
        domain = context.get("domain", "general")
        complexity_score = context.get("complexity_score", 0.5)
        
        # Select base approach
        if complexity_score > 0.7:
            base_approach = {
                "scamper": True,
                "biomimicry": True,
                "lateral_thinking": True,
                "design_thinking": True
            }
        elif complexity_score > 0.4:
            base_approach = {
                "scamper": True,
                "biomimicry": False,
                "lateral_thinking": True,
                "design_thinking": True
            }
        else:
            base_approach = {
                "scamper": True,
                "biomimicry": False,
                "lateral_thinking": False,
                "design_thinking": True
            }
        
        # Customize based on domain
        if domain in ["technology", "engineering"]:
            base_approach["biomimicry"] = True
        elif domain in ["business", "marketing"]:
            base_approach["design_thinking"] = True
        
        return base_approach
    
    async def _generate_creative_concepts(
        self, 
        query: str, 
        findings: List[Any], 
        approach: Dict[str, Any]
    ) -> List[CreativeConcept]:
        """Generate creative concepts using multiple methods"""
        self.log_activity("Generating creative concepts", {"approach": approach})
        
        concepts = []
        
        # Generate concepts using each enabled method
        if approach.get("scamper"):
            scamper_concepts = await self._generate_scamper_concepts(query, findings)
            concepts.extend(scamper_concepts)
        
        if approach.get("biomimicry"):
            biomimicry_concepts = await self._generate_biomimicry_concepts(query, findings)
            concepts.extend(biomimicry_concepts)
        
        if approach.get("lateral_thinking"):
            lateral_concepts = await self._generate_lateral_thinking_concepts(query, findings)
            concepts.extend(lateral_concepts)
        
        if approach.get("design_thinking"):
            design_concepts = await self._generate_design_thinking_concepts(query, findings)
            concepts.extend(design_concepts)
        
        # Remove duplicates and rank by innovation score
        unique_concepts = self._deduplicate_concepts(concepts)
        ranked_concepts = self._rank_concepts_by_innovation(unique_concepts)
        
        return ranked_concepts[:10]  # Return top 10 concepts
    
    async def _generate_scamper_concepts(self, query: str, findings: List[Any]) -> List[CreativeConcept]:
        """Generate concepts using SCAMPER method"""
        concepts = []
        
        for method, question in self.creativity_methods["scamper"].items():
            prompt = f"""
            Apply the SCAMPER method to this query:
            
            Query: {query}
            SCAMPER Question: {question}
            
            Generate 2-3 creative concepts that answer this question.
            Focus on practical, innovative solutions.
            """
            
            response = await self.generate_response_with_gemini(prompt)
            
            if response.get("success"):
                concept_titles = self._extract_concept_titles(response["content"])
                
                for title in concept_titles:
                    concept = CreativeConcept(
                        title=title,
                        description=response["content"],
                        innovation_score=self._calculate_innovation_score(title, method),
                        feasibility_score=self._calculate_feasibility_score(title),
                        implementation_path=self._generate_implementation_path(title),
                        alternatives=self._generate_alternatives(title),
                        inspiration_sources=[f"SCAMPER: {method}"],
                        constraints=self._identify_constraints(title)
                    )
                    concepts.append(concept)
        
        return concepts
    
    async def _generate_biomimicry_concepts(self, query: str, findings: List[Any]) -> List[CreativeConcept]:
        """Generate concepts using biomimicry"""
        concepts = []
        
        prompt = f"""
        Apply biomimicry to solve this problem:
        
        Query: {query}
        
        Look to nature for inspiration:
        1. How does nature solve similar problems?
        2. What biological systems could inspire solutions?
        3. What evolutionary principles apply?
        
        Generate 3-4 nature-inspired concepts.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            concept_titles = self._extract_concept_titles(response["content"])
            
            for title in concept_titles:
                concept = CreativeConcept(
                    title=title,
                    description=response["content"],
                    innovation_score=self._calculate_innovation_score(title, "biomimicry"),
                    feasibility_score=self._calculate_feasibility_score(title),
                    implementation_path=self._generate_implementation_path(title),
                    alternatives=self._generate_alternatives(title),
                    inspiration_sources=["Biomimicry", "Nature-inspired design"],
                    constraints=self._identify_constraints(title)
                )
                concepts.append(concept)
        
        return concepts
    
    async def _generate_lateral_thinking_concepts(self, query: str, findings: List[Any]) -> List[CreativeConcept]:
        """Generate concepts using lateral thinking"""
        concepts = []
        
        # Random stimulation
        random_words = self._generate_random_words()
        
        prompt = f"""
        Use lateral thinking to solve this problem:
        
        Query: {query}
        Random stimulation words: {random_words}
        
        Use these random words to stimulate creative thinking.
        Challenge assumptions and think outside the box.
        
        Generate 2-3 unconventional concepts.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            concept_titles = self._extract_concept_titles(response["content"])
            
            for title in concept_titles:
                concept = CreativeConcept(
                    title=title,
                    description=response["content"],
                    innovation_score=self._calculate_innovation_score(title, "lateral_thinking"),
                    feasibility_score=self._calculate_feasibility_score(title),
                    implementation_path=self._generate_implementation_path(title),
                    alternatives=self._generate_alternatives(title),
                    inspiration_sources=["Lateral thinking", "Random stimulation"],
                    constraints=self._identify_constraints(title)
                )
                concepts.append(concept)
        
        return concepts
    
    async def _generate_design_thinking_concepts(self, query: str, findings: List[Any]) -> List[CreativeConcept]:
        """Generate concepts using design thinking"""
        concepts = []
        
        prompt = f"""
        Apply design thinking to this problem:
        
        Query: {query}
        
        Follow the design thinking process:
        1. Empathize: Understand user needs
        2. Define: Define the problem clearly
        3. Ideate: Generate many ideas
        4. Prototype: Create quick prototypes
        5. Test: Test and iterate
        
        Generate 3-4 user-centered concepts.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            concept_titles = self._extract_concept_titles(response["content"])
            
            for title in concept_titles:
                concept = CreativeConcept(
                    title=title,
                    description=response["content"],
                    innovation_score=self._calculate_innovation_score(title, "design_thinking"),
                    feasibility_score=self._calculate_feasibility_score(title),
                    implementation_path=self._generate_implementation_path(title),
                    alternatives=self._generate_alternatives(title),
                    inspiration_sources=["Design thinking", "User-centered design"],
                    constraints=self._identify_constraints(title)
                )
                concepts.append(concept)
        
        return concepts
    
    def _deduplicate_concepts(self, concepts: List[CreativeConcept]) -> List[CreativeConcept]:
        """Remove duplicate concepts based on title similarity"""
        unique_concepts = []
        seen_titles = set()
        
        for concept in concepts:
            title_lower = concept.title.lower()
            
            # Check for similarity with existing concepts
            is_duplicate = False
            for seen_title in seen_titles:
                if self._calculate_similarity(title_lower, seen_title) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_concepts.append(concept)
                seen_titles.add(title_lower)
        
        return unique_concepts
    
    def _rank_concepts_by_innovation(self, concepts: List[CreativeConcept]) -> List[CreativeConcept]:
        """Rank concepts by innovation score"""
        return sorted(concepts, key=lambda x: x.innovation_score, reverse=True)
    
    def _calculate_innovation_score(self, title: str, method: str) -> float:
        """Calculate innovation score for a concept"""
        score = 0.5  # Base score
        
        # Method-specific scoring
        if method == "biomimicry":
            score += 0.2
        elif method == "lateral_thinking":
            score += 0.15
        elif method == "design_thinking":
            score += 0.1
        
        # Title-based scoring
        innovation_keywords = ["novel", "innovative", "revolutionary", "breakthrough", "disruptive"]
        for keyword in innovation_keywords:
            if keyword in title.lower():
                score += 0.1
        
        # Random variation for diversity
        score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_feasibility_score(self, title: str) -> float:
        """Calculate feasibility score for a concept"""
        score = 0.6  # Base feasibility
        
        # Adjust based on complexity indicators
        complexity_indicators = ["complex", "advanced", "sophisticated", "cutting-edge"]
        for indicator in complexity_indicators:
            if indicator in title.lower():
                score -= 0.1
        
        # Adjust based on simplicity indicators
        simplicity_indicators = ["simple", "basic", "straightforward", "practical"]
        for indicator in simplicity_indicators:
            if indicator in title.lower():
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_implementation_path(self, title: str) -> List[str]:
        """Generate implementation path for a concept"""
        steps = [
            "Research and analysis",
            "Concept development",
            "Prototype creation",
            "Testing and validation",
            "Refinement and optimization",
            "Implementation and deployment"
        ]
        
        # Customize based on concept type
        if "technology" in title.lower():
            steps.insert(2, "Technical feasibility study")
        elif "business" in title.lower():
            steps.insert(2, "Market analysis")
        
        return steps
    
    def _generate_alternatives(self, title: str) -> List[str]:
        """Generate alternative approaches for a concept"""
        alternatives = []
        
        # Generate variations based on concept type
        if "digital" in title.lower():
            alternatives.extend(["Analog approach", "Hybrid solution", "Mobile-first design"])
        elif "physical" in title.lower():
            alternatives.extend(["Digital twin", "Virtual implementation", "Remote solution"])
        else:
            alternatives.extend(["Alternative A", "Alternative B", "Alternative C"])
        
        return alternatives
    
    def _identify_constraints(self, title: str) -> List[str]:
        """Identify potential constraints for a concept"""
        constraints = []
        
        # Common constraints
        constraints.extend(["Time limitations", "Budget constraints", "Technical feasibility"])
        
        # Domain-specific constraints
        if "technology" in title.lower():
            constraints.extend(["Infrastructure requirements", "Compatibility issues"])
        elif "business" in title.lower():
            constraints.extend(["Market acceptance", "Competition", "Regulatory compliance"])
        
        return constraints
    
    def _calculate_innovation_metrics(self, concepts: List[CreativeConcept]) -> Dict[str, float]:
        """Calculate innovation metrics for the concepts"""
        if not concepts:
            return {
                "average_innovation_score": 0.0,
                "average_feasibility_score": 0.0,
                "innovation_diversity": 0.0,
                "breakthrough_potential": 0.0
            }
        
        # Calculate averages
        avg_innovation = sum(c.innovation_score for c in concepts) / len(concepts)
        avg_feasibility = sum(c.feasibility_score for c in concepts) / len(concepts)
        
        # Calculate diversity
        innovation_scores = [c.innovation_score for c in concepts]
        diversity = max(innovation_scores) - min(innovation_scores)
        
        # Calculate breakthrough potential
        breakthrough_count = sum(1 for c in concepts if c.innovation_score > 0.8)
        breakthrough_potential = breakthrough_count / len(concepts)
        
        return {
            "average_innovation_score": avg_innovation,
            "average_feasibility_score": avg_feasibility,
            "innovation_diversity": diversity,
            "breakthrough_potential": breakthrough_potential
        }
    
    def _determine_synthesis_approach(self, concepts: List[CreativeConcept], approach: Dict[str, Any]) -> str:
        """Determine the best synthesis approach for the concepts"""
        if not concepts:
            return "standard"
        
        # Analyze concept characteristics
        high_innovation_count = sum(1 for c in concepts if c.innovation_score > 0.7)
        high_feasibility_count = sum(1 for c in concepts if c.feasibility_score > 0.7)
        
        if high_innovation_count > len(concepts) * 0.5:
            return "innovation_focused"
        elif high_feasibility_count > len(concepts) * 0.5:
            return "practical_focused"
        else:
            return "balanced"
    
    async def _generate_cross_domain_insights(self, concepts: List[CreativeConcept], query: str) -> List[str]:
        """Generate cross-domain insights from concepts"""
        if not concepts:
            return []
        
        prompt = f"""
        Generate cross-domain insights from these creative concepts:
        
        Query: {query}
        Concepts: {[c.title for c in concepts]}
        
        Look for:
        1. Connections between different domains
        2. Unexpected applications
        3. Transferable principles
        4. Synergistic combinations
        5. Novel perspectives
        
        Provide 3-5 cross-domain insights.
        """
        
        response = await self.generate_response_with_gemini(prompt)
        
        if response.get("success"):
            insights = self._extract_insights_from_text(response["content"])
            return insights[:5]  # Limit to 5 insights
        else:
            return ["Cross-domain analysis completed", "Multiple perspectives identified"]
    
    def _calculate_creative_confidence(self, concepts: List[CreativeConcept], metrics: Dict[str, float]) -> float:
        """Calculate confidence score for creative output"""
        if not concepts:
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Concept quality
        avg_innovation = metrics.get("average_innovation_score", 0.0)
        avg_feasibility = metrics.get("average_feasibility_score", 0.0)
        
        confidence += avg_innovation * 0.3
        confidence += avg_feasibility * 0.2
        
        # Diversity bonus
        diversity = metrics.get("innovation_diversity", 0.0)
        confidence += diversity * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    # Helper methods
    def _extract_concept_titles(self, content: str) -> List[str]:
        """Extract concept titles from content"""
        titles = []
        
        # Look for numbered or bulleted items
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Extract title after number/bullet
                title = line.split(' ', 1)[1] if ' ' in line else line
                titles.append(title)
        
        # If no structured titles found, create from content
        if not titles:
            sentences = content.split('.')
            for sentence in sentences[:3]:  # Take first 3 sentences
                sentence = sentence.strip()
                if len(sentence) > 10:
                    titles.append(sentence[:50] + "..." if len(sentence) > 50 else sentence)
        
        return titles[:5]  # Limit to 5 titles
    
    def _generate_random_words(self) -> List[str]:
        """Generate random words for lateral thinking stimulation"""
        random_words = [
            "butterfly", "mountain", "ocean", "forest", "city", "village", "bridge", "river",
            "cloud", "star", "moon", "sun", "wind", "rain", "snow", "fire", "earth",
            "metal", "wood", "stone", "glass", "plastic", "fabric", "paper", "ink"
        ]
        
        return random.sample(random_words, 5)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _extract_insights_from_text(self, content: str) -> List[str]:
        """Extract insights from text content"""
        insights = []
        
        # Split into sentences
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in ["insight", "connection", "pattern", "principle"]):
                insights.append(sentence)
        
        return insights[:5]  # Limit to 5 insights
    
    # Query-specific methods
    async def _generate_ideas(self, query_data: Dict[str, Any]) -> Optional[CreativeOutput]:
        """Generate creative ideas for a specific query"""
        query = query_data.get("query", "")
        
        if not query:
            return None
        
        concepts = await self._generate_creative_concepts(query, [], {"scamper": True, "design_thinking": True})
        metrics = self._calculate_innovation_metrics(concepts)
        
        return CreativeOutput(
            agent_type=AgentType.CREATIVE,
            content={"concepts": [c.dict() for c in concepts]},
            confidence_score=0.8,
            concepts=concepts,
            innovation_metrics=metrics,
            synthesis_approach="standard",
            cross_domain_insights=[]
        )
    
    async def _solve_problem_creatively(self, query_data: Dict[str, Any]) -> Optional[CreativeOutput]:
        """Solve a problem using creative approaches"""
        problem = query_data.get("problem", "")
        
        if not problem:
            return None
        
        concepts = await self._generate_creative_concepts(problem, [], {"scamper": True, "lateral_thinking": True})
        metrics = self._calculate_innovation_metrics(concepts)
        
        return CreativeOutput(
            agent_type=AgentType.CREATIVE,
            content={"concepts": [c.dict() for c in concepts]},
            confidence_score=0.7,
            concepts=concepts,
            innovation_metrics=metrics,
            synthesis_approach="problem_solving",
            cross_domain_insights=[]
        )
    
    async def _analyze_innovation_potential(self, query_data: Dict[str, Any]) -> Optional[CreativeOutput]:
        """Analyze innovation potential of existing concepts"""
        concepts_data = query_data.get("concepts", [])
        
        if not concepts_data:
            return None
        
        # Convert to CreativeConcept objects
        concepts = []
        for concept_data in concepts_data:
            concept = CreativeConcept(
                title=concept_data.get("title", ""),
                description=concept_data.get("description", ""),
                innovation_score=concept_data.get("innovation_score", 0.5),
                feasibility_score=concept_data.get("feasibility_score", 0.5)
            )
            concepts.append(concept)
        
        metrics = self._calculate_innovation_metrics(concepts)
        
        return CreativeOutput(
            agent_type=AgentType.CREATIVE,
            content={"analysis": "Innovation potential analyzed"},
            confidence_score=0.9,
            concepts=concepts,
            innovation_metrics=metrics,
            synthesis_approach="analysis",
            cross_domain_insights=[]
        ) 