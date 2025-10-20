"""
Gemini API integration for professor dashboard summaries
"""
import os
import json
import httpx
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


class GeminiService:
    """
    Service for calling Google Gemini API to generate summaries
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("⚠️  Warning: GEMINI_API_KEY not set")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    async def generate_summary(self, feedback_data: List[Dict]) -> str:
        """
        Generate a summary of student feedback using Gemini API
        
        Args:
            feedback_data: List of low-rated Q&A pairs with feedback
            
        Returns:
            Gemini-generated summary text
        """
        if not self.api_key:
            return "Error: Gemini API key not configured. Please set GEMINI_API_KEY in environment variables."
        
        # Build prompt for Gemini
        prompt = self._build_summary_prompt(feedback_data)
        
        # Call Gemini API
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [{
                                "text": prompt
                            }]
                        }],
                        "generationConfig": {
                            "temperature": 0.4,
                            "topK": 32,
                            "topP": 1,
                            "maxOutputTokens": 2048,
                        }
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        text = result["candidates"][0]["content"]["parts"][0]["text"]
                        return text
                    else:
                        return "Error: No response generated from Gemini API"
                else:
                    return f"Error: Gemini API returned status {response.status_code}: {response.text}"
                    
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"

    def _build_summary_prompt(self, feedback_data: List[Dict]) -> str:
        """
        Build a prompt for Gemini to analyze student feedback
        
        Args:
            feedback_data: List of feedback entries
            
        Returns:
            Formatted prompt string
        """
        prompt = """You are an educational data analyst helping a professor improve their Data Structures and Algorithms (DSA) course.

Below are student questions that received low satisfaction ratings (thumbs down) or negative feedback. Your task is to:

1. Identify the top 5 common topics or concepts where students are struggling
2. For each topic, provide:
   - The concept/topic name
   - Common misunderstandings or issues
   - 2-3 example student questions
   - Specific teaching recommendations

Use clear, actionable bullet points. Focus on patterns and trends, not individual cases.

---

STUDENT FEEDBACK DATA:

"""
        
        for i, item in enumerate(feedback_data[:50], 1):  # Limit to 50 to avoid token limits
            prompt += f"\n{i}. Question: {item['message']}\n"
            prompt += f"   Answer: {item['response'][:200]}...\n"  # Truncate long responses
            if item.get('comment'):
                prompt += f"   Student Comment: {item['comment']}\n"
            prompt += "\n"
        
        prompt += """
---

Please provide your analysis in the following format:

## Summary of Common Issues

### Topic 1: [Topic Name]
- **Misunderstandings:** [Description]
- **Example Questions:**
  - "[Question 1]"
  - "[Question 2]"
- **Teaching Recommendations:**
  - [Recommendation 1]
  - [Recommendation 2]

[Continue for topics 2-5]

## Overall Recommendations
[General suggestions for improving the course]
"""
        
        return prompt


# Global instance
gemini_service = GeminiService()


def get_gemini_service() -> GeminiService:
    """
    Get the global Gemini service instance
    """
    return gemini_service

