#!/usr/bin/env python3
"""
Example script demonstrating the Multi-Agent Autonomous Research System
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

from system_orchestrator import SystemOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_creative_example():
    """Run a creative research example"""
    print("🎨 Creative Research Example")
    print("=" * 50)
    
    orchestrator = SystemOrchestrator()
    await orchestrator.start_system()
    
    query = "Design a new natural language with 5 letters"
    
    print(f"Query: {query}")
    print("This demonstrates creative problem-solving capabilities...")
    
    result = await orchestrator.execute_research_query(query, {
        "domain": "linguistics",
        "research_depth": "comprehensive"
    })
    
    # Save result
    filename = f"creative_example_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"Results saved to: {filename}")
    
    # Show summary
    if result.get("status") == "completed":
        print("\n📊 Creative Research Summary:")
        results = result.get("results", {})
        
        if "creative" in results:
            concepts = results["creative"].get("concepts", [])
            print(f"Creative Concepts Generated: {len(concepts)}")
            for i, concept in enumerate(concepts[:3], 1):
                print(f"  {i}. {concept.get('title', 'Untitled')}")
        
        if "documentation" in results:
            print("Documentation: ✅ Generated")
    
    await orchestrator.stop_system()
    return result

async def run_analytical_example():
    """Run an analytical research example"""
    print("\n📈 Analytical Research Example")
    print("=" * 50)
    
    orchestrator = SystemOrchestrator()
    await orchestrator.start_system()
    
    query = "What are the economic implications of vertical farming in urban environments?"
    
    print(f"Query: {query}")
    print("This demonstrates comprehensive analysis capabilities...")
    
    result = await orchestrator.execute_research_query(query, {
        "domain": "economics",
        "research_depth": "comprehensive"
    })
    
    # Save result
    filename = f"analytical_example_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"Results saved to: {filename}")
    
    # Show summary
    if result.get("status") == "completed":
        print("\n📊 Analytical Research Summary:")
        results = result.get("results", {})
        
        if "research" in results:
            findings = results["research"].get("findings", [])
            print(f"Research Findings: {len(findings)}")
        
        if "analysis" in results:
            insights = results["analysis"].get("insights", [])
            print(f"Key Insights: {len(insights)}")
            for i, insight in enumerate(insights[:3], 1):
                print(f"  {i}. {insight[:100]}...")
        
        if "validation" in results:
            validation = results["validation"].get("validation_report", {})
            confidence = validation.get("overall_confidence", 0.0)
            print(f"Validation Confidence: {confidence:.2f}")
    
    await orchestrator.stop_system()
    return result

async def run_technical_example():
    """Run a technical research example"""
    print("\n🔬 Technical Research Example")
    print("=" * 50)
    
    orchestrator = SystemOrchestrator()
    await orchestrator.start_system()
    
    query = "Analyze the impact of artificial intelligence on healthcare delivery"
    
    print(f"Query: {query}")
    print("This demonstrates technical analysis capabilities...")
    
    result = await orchestrator.execute_research_query(query, {
        "domain": "technology",
        "research_depth": "comprehensive"
    })
    
    # Save result
    filename = f"technical_example_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"Results saved to: {filename}")
    
    # Show summary
    if result.get("status") == "completed":
        print("\n📊 Technical Research Summary:")
        results = result.get("results", {})
        
        if "research" in results:
            sources = results["research"].get("sources", [])
            print(f"Sources Analyzed: {len(sources)}")
        
        if "analysis" in results:
            patterns = results["analysis"].get("patterns", [])
            correlations = results["analysis"].get("correlations", [])
            print(f"Patterns Identified: {len(patterns)}")
            print(f"Correlations Found: {len(correlations)}")
        
        if "documentation" in results:
            doc_content = results["documentation"].get("structured_content", {})
            sections = list(doc_content.keys())
            print(f"Documentation Sections: {len(sections)}")
    
    await orchestrator.stop_system()
    return result

async def run_system_demo():
    """Run a comprehensive system demonstration"""
    print("🚀 Autodisc - Multi-Agent Research System Demo")
    print("=" * 60)
    print()
    
    # Show system capabilities
    print("System Capabilities:")
    print("• Multi-agent coordination")
    print("• Comprehensive research")
    print("• Deep analysis")
    print("• Creative problem-solving")
    print("• Quality validation")
    print("• Structured documentation")
    print()
    
    # Run examples
    try:
        # Creative example
        await run_creative_example()
        
        # Analytical example
        await run_analytical_example()
        
        # Technical example
        await run_technical_example()
        
        print("\n✅ Demo completed successfully!")
        print("Check the generated JSON files for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        logger.error(f"Demo error: {str(e)}")

async def run_quick_test():
    """Run a quick test to verify system functionality"""
    print("🧪 Quick System Test")
    print("=" * 30)
    
    orchestrator = SystemOrchestrator()
    
    try:
        await orchestrator.start_system()
        
        # Test system status
        status = await orchestrator.get_system_status()
        print(f"System Status: {status['system_status']}")
        print(f"Active Agents: {len(status['agents'])}")
        
        # Test simple query
        query = "What is artificial intelligence?"
        print(f"\nTesting simple query: {query}")
        
        result = await orchestrator.execute_research_query(query, {
            "research_depth": "quick"
        })
        
        print(f"Test Status: {result.get('status')}")
        
        if result.get("status") == "completed":
            print("✅ System test passed!")
        else:
            print("❌ System test failed!")
        
        await orchestrator.stop_system()
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        await orchestrator.stop_system()

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "creative":
            asyncio.run(run_creative_example())
        elif mode == "analytical":
            asyncio.run(run_analytical_example())
        elif mode == "technical":
            asyncio.run(run_technical_example())
        elif mode == "test":
            asyncio.run(run_quick_test())
        else:
            print("Usage: python example.py [creative|analytical|technical|test|demo]")
    else:
        # Run full demo
        asyncio.run(run_system_demo())

if __name__ == "__main__":
    main() 