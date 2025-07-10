#!/usr/bin/env python3
"""
Main entry point for the Multi-Agent Autonomous Research System
"""
import asyncio
import argparse
import json
import logging
import sys
from typing import Dict, Any, Optional

from config import Config
from system_orchestrator import SystemOrchestrator

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('autodisc.log')
    ]
)

logger = logging.getLogger(__name__)

class AutodiscCLI:
    """Command-line interface for the Autodisc research system"""
    
    def __init__(self):
        """Initialize the CLI"""
        self.orchestrator: Optional[SystemOrchestrator] = None
        
    async def start(self):
        """Start the CLI interface"""
        try:
            # Validate configuration
            Config.validate_config()
            
            # Initialize orchestrator
            self.orchestrator = SystemOrchestrator()
            
            # Start the system
            await self.orchestrator.start_system()
            
            logger.info("Autodisc CLI started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start CLI: {str(e)}")
            sys.exit(1)
    
    async def stop(self):
        """Stop the CLI interface"""
        if self.orchestrator:
            await self.orchestrator.stop_system()
            logger.info("Autodisc CLI stopped")
    
    async def execute_query(self, query: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a research query"""
        if not self.orchestrator:
            raise Exception("System not initialized")
        
        return await self.orchestrator.execute_research_query(query, options)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        if not self.orchestrator:
            raise Exception("System not initialized")
        
        return await self.orchestrator.get_system_status()
    
    async def list_sessions(self) -> list:
        """List active sessions"""
        if not self.orchestrator:
            raise Exception("System not initialized")
        
        return await self.orchestrator.list_active_sessions()

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Autodisc - Multi-Agent Autonomous Research System")
    parser.add_argument("--query", "-q", help="Research query to execute")
    parser.add_argument("--domain", help="Domain of the research (e.g., technology, business, science)")
    parser.add_argument("--depth", choices=["quick", "standard", "comprehensive"], 
                       default="comprehensive", help="Research depth")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--sessions", action="store_true", help="List active sessions")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    cli = AutodiscCLI()
    
    try:
        await cli.start()
        
        if args.status:
            # Show system status
            status = await cli.get_status()
            print(json.dumps(status, indent=2, default=str))
            return
        
        if args.sessions:
            # List active sessions
            sessions = await cli.list_sessions()
            print(json.dumps(sessions, indent=2, default=str))
            return
        
        if args.query:
            # Execute a specific query
            options = {}
            if args.domain:
                options["domain"] = args.domain
            if args.depth:
                options["research_depth"] = args.depth
            
            print(f"Executing research query: {args.query}")
            print("This may take several minutes...")
            
            result = await cli.execute_query(args.query, options)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Results saved to {args.output}")
            else:
                print(json.dumps(result, indent=2, default=str))
            
            return
        
        if args.interactive:
            # Start interactive mode
            await interactive_mode(cli)
            return
        
        # Show help if no arguments provided
        parser.print_help()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)
    finally:
        await cli.stop()

async def interactive_mode(cli: AutodiscCLI):
    """Interactive mode for the CLI"""
    print("Autodisc Interactive Mode")
    print("Type 'help' for commands, 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            command = input("autodisc> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            if command.lower() == 'help':
                print_help()
                continue
            
            if command.lower() == 'status':
                status = await cli.get_status()
                print(json.dumps(status, indent=2, default=str))
                continue
            
            if command.lower() == 'sessions':
                sessions = await cli.list_sessions()
                print(json.dumps(sessions, indent=2, default=str))
                continue
            
            if command.startswith('query:'):
                # Execute a query
                query = command[6:].strip()
                if query:
                    print(f"Executing: {query}")
                    print("This may take several minutes...")
                    
                    result = await cli.execute_query(query)
                    
                    # Save to file with timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"research_result_{timestamp}.json"
                    
                    with open(filename, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    
                    print(f"Research completed! Results saved to {filename}")
                    
                    # Show summary
                    if result.get("status") == "completed":
                        print("\nResearch Summary:")
                        print(f"Session ID: {result.get('session_id')}")
                        print(f"Status: {result.get('status')}")
                        
                        results = result.get("results", {})
                        if "research" in results:
                            findings = results["research"].get("findings", [])
                            print(f"Findings: {len(findings)}")
                        
                        if "creative" in results:
                            concepts = results["creative"].get("concepts", [])
                            print(f"Creative Concepts: {len(concepts)}")
                        
                        if "documentation" in results:
                            print("Documentation: Generated")
                    
                    continue
            
            print(f"Unknown command: {command}")
            print("Type 'help' for available commands")
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def print_help():
    """Print help information"""
    print("Available commands:")
    print("  help                    - Show this help")
    print("  status                  - Show system status")
    print("  sessions                - List active sessions")
    print("  query: <your query>     - Execute a research query")
    print("  quit/exit/q             - Exit interactive mode")
    print()
    print("Example queries:")
    print("  query: What are the economic implications of vertical farming in urban environments?")
    print("  query: Design a new natural language with 5 letters")
    print("  query: Analyze the impact of artificial intelligence on healthcare")
    print()

async def run_example():
    """Run an example research query"""
    print("Running example research query...")
    
    cli = AutodiscCLI()
    
    try:
        await cli.start()
        
        # Example query
        query = "What are the economic implications of vertical farming in urban environments?"
        
        print(f"Executing: {query}")
        print("This demonstrates the full multi-agent research workflow...")
        
        result = await cli.execute_query(query, {
            "domain": "economics",
            "research_depth": "comprehensive"
        })
        
        # Save result
        with open("example_result.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print("Example completed! Check example_result.json for results.")
        
        # Show summary
        if result.get("status") == "completed":
            print("\nExample Research Summary:")
            print(f"Session ID: {result.get('session_id')}")
            
            results = result.get("results", {})
            if "research" in results:
                findings = results["research"].get("findings", [])
                print(f"Research Findings: {len(findings)}")
            
            if "analysis" in results:
                print("Analysis: Completed")
            
            if "creative" in results:
                concepts = results["creative"].get("concepts", [])
                print(f"Creative Concepts: {len(concepts)}")
            
            if "validation" in results:
                print("Validation: Completed")
            
            if "documentation" in results:
                print("Documentation: Generated")
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
    finally:
        await cli.stop()

if __name__ == "__main__":
    # Check if running example
    if len(sys.argv) == 1:
        print("Autodisc - Multi-Agent Autonomous Research System")
        print("=" * 60)
        print()
        print("No arguments provided. Running example...")
        print()
        asyncio.run(run_example())
    else:
        asyncio.run(main()) 