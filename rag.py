from llm import GemmaLLM
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class NCORAG:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = GemmaLLM()
        
    def build_context(self, query: str, top_k: int) -> tuple[str, List[Dict]]:
        """
        Build context from retrieved documents
        
        Returns:
            Tuple of (context_string, documents_list)
        """
        try:
            docs = self.retriever.search(query, top_k)
            
            if not docs:
                logger.warning("No documents retrieved for query")
                return "", []
            
            context_blocks = []
            for i, d in enumerate(docs, 1):
                # Sanitize and format each document
                title = d.get('title', 'Unknown').strip()
                nco_code = d.get('nco_2015', 'N/A').strip()
                description = d.get('description', 'No description available').strip()
                
                block = f"""
OCCUPATION {i}:
Title: {title}
NCO Code: {nco_code}
Description: {description}
"""
                context_blocks.append(block)
            
            context = "\n".join(context_blocks)
            logger.info(f"Built context with {len(docs)} documents")
            
            return context, docs
            
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return "", []
    
    def generate_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate answer based on retrieved context
        
        Args:
            query: User's query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        # Build context from retrieved documents
        context, docs = self.build_context(query, top_k)
        
        if not docs:
            return {
                "answer": "No relevant occupations found. Please try a different description.",
                "sources": []
            }
        
        # Create structured prompt
        prompt = f"""You are an expert career advisor specializing in the NCO 2015 (National Classification of Occupations) system.

IMPORTANT RULES:
- ONLY use information from the provided occupations below
- Do NOT invent or assume additional information
- If information is not in the data, say "Not specified in NCO data"
- Be helpful, specific, and professional

USER QUERY:
"{query}"

OCCUPATIONS FROM NCO 2015 DATABASE:
{context}

Based SOLELY on the occupations above, provide a comprehensive career guidance response with these sections:

1. TOP MATCHING OCCUPATIONS
   List the most relevant occupations and their NCO codes.

2. WHY THEY MATCH
   For each top match, explain specific alignment with the query.

3. KEY SKILLS REQUIRED
   Extract skills mentioned in the descriptions.

4. CAREER OUTLOOK
   Based on available information, provide insights about demand and opportunities.

5. NEXT STEPS
   Suggest how to pursue these careers (training, education, certifications).

Format your response clearly with emojis and section headers for readability."""

        # Generate response
        logger.info(f"Generating answer for query: {query[:50]}...")
        answer = self.llm.generate(prompt, max_tokens=1000)
        
        # If LLM fails, provide fallback response
        if answer.startswith("Error:"):
            # Create a simple fallback response using the retrieved docs
            fallback = "**Retrieved Occupations:**\n\n"
            for i, doc in enumerate(docs[:3], 1):
                fallback += f"{i}. **{doc.get('title', 'Unknown')}** (NCO: {doc.get('nco_2015', 'N/A')})\n"
                fallback += f"   {doc.get('description', '')[:200]}...\n\n"
            fallback += "\n*Note: AI generation failed. Please check Ollama connection.*"
            answer = fallback
        
        return {
            "answer": answer,
            "sources": docs
        }
    
    def generate_structured_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate a more structured answer with parsed sections
        """
        result = self.generate_answer(query, top_k)
        
        # You could add post-processing here to parse the response
        # into structured sections if needed
        
        return result