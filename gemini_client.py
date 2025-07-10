"""
Gemini API client wrapper for the Multi-Agent Autonomous Research System
"""
import google.generativeai as genai
import asyncio
import time
from typing import Dict, List, Any, Optional
from config import Config
import logging

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google's Gemini API"""
    
    def __init__(self):
        """Initialize the Gemini client"""
        self.api_key = Config.GEMINI_API_KEY
        self.model = Config.GEMINI_MODEL
        self.max_tokens = Config.GEMINI_MAX_TOKENS
        self.temperature = Config.GEMINI_TEMPERATURE
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model_instance = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature
            )
        )
        
        logger.info(f"Gemini client initialized with model: {self.model}")
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a response from Gemini API
        
        Args:
            prompt: The main prompt to send
            context: Additional context information
            system_prompt: System-level instructions
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        # Construct the full prompt
        full_prompt = self._construct_prompt(prompt, context, system_prompt)
        
        for attempt in range(max_retries):
            try:
                # Generate response
                response = await self._make_api_call(full_prompt)
                
                processing_time = time.time() - start_time
                
                return {
                    "content": response.text,
                    "confidence_score": self._calculate_confidence(response),
                    "processing_time": processing_time,
                    "attempts": attempt + 1,
                    "success": True
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries - 1:
                    processing_time = time.time() - start_time
                    return {
                        "content": f"Error: {str(e)}",
                        "confidence_score": 0.0,
                        "processing_time": processing_time,
                        "attempts": attempt + 1,
                        "success": False,
                        "error": str(e)
                    }
                
                # Wait before retrying
                await asyncio.sleep(2 ** attempt)
    
    async def _make_api_call(self, prompt: str) -> Any:
        """Make the actual API call to Gemini"""
        # For now, we'll use synchronous call since async support might be limited
        # In production, you might want to use a proper async wrapper
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: self.model_instance.generate_content(prompt)
        )
        return response
    
    def _construct_prompt(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Construct the full prompt with context and system instructions"""
        parts = []
        
        if system_prompt:
            parts.append(f"System Instructions: {system_prompt}\n")
        
        if context:
            parts.append(f"Context: {context}\n")
        
        parts.append(f"Task: {prompt}")
        
        return "\n".join(parts)
    
    def _calculate_confidence(self, response: Any) -> float:
        """Calculate confidence score based on response characteristics"""
        # This is a simplified confidence calculation
        # In a real implementation, you might use more sophisticated methods
        
        confidence = 0.8  # Base confidence
        
        # Adjust based on response length (longer responses might be more detailed)
        if hasattr(response, 'text'):
            text_length = len(response.text)
            if text_length > 1000:
                confidence += 0.1
            elif text_length < 100:
                confidence -= 0.2
        
        # Adjust based on response structure
        if hasattr(response, 'text') and response.text:
            if any(keyword in response.text.lower() for keyword in ['however', 'although', 'but', 'nevertheless']):
                confidence -= 0.1  # Uncertainty indicators
        
        return max(0.0, min(1.0, confidence))
    
    async def generate_structured_response(
        self, 
        prompt: str, 
        output_format: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured response in a specific format
        
        Args:
            prompt: The main prompt
            output_format: Dictionary describing the desired output structure
            context: Additional context
            
        Returns:
            Structured response matching the specified format
        """
        format_instructions = self._create_format_instructions(output_format)
        
        structured_prompt = f"""
{prompt}

Please provide your response in the following structured format:

{format_instructions}

Ensure all fields are properly filled and the response is valid JSON.
"""
        
        response = await self.generate_response(structured_prompt, context)
        
        if response["success"]:
            # Try to parse as JSON if possible
            try:
                import json
                # Extract JSON from the response if it's wrapped in markdown
                content = response["content"]
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    json_content = content[start:end].strip()
                else:
                    json_content = content
                
                structured_data = json.loads(json_content)
                response["structured_data"] = structured_data
            except json.JSONDecodeError:
                response["structured_data"] = None
                response["confidence_score"] *= 0.8  # Reduce confidence for parsing failure
        
        return response
    
    def _create_format_instructions(self, output_format: Dict[str, Any]) -> str:
        """Create format instructions for structured output"""
        import json
        return f"""
```json
{json.dumps(output_format, indent=2)}
```
"""
    
    async def batch_generate(
        self, 
        prompts: List[str], 
        contexts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts in parallel
        
        Args:
            prompts: List of prompts to process
            contexts: Optional list of contexts for each prompt
            
        Returns:
            List of response dictionaries
        """
        if contexts is None:
            contexts = [None] * len(prompts)
        
        tasks = [
            self.generate_response(prompt, context)
            for prompt, context in zip(prompts, contexts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "content": f"Error: {str(result)}",
                    "confidence_score": 0.0,
                    "processing_time": 0.0,
                    "attempts": 1,
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key_configured": bool(self.api_key)
        } 