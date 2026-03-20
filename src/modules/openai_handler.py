"""
OpenAI Handler Module
=====================
Manages OpenAI API interactions for embeddings and completions.

This module handles:
- OpenAI API authentication
- Embedding generation
- GPT completions for Q&A
- Error handling and retry logic
- API call logging
"""

import os
import time
import logging
from typing import List, Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIHandler:
    """
    A class to handle OpenAI API operations.
    
    Attributes:
        client: OpenAI client instance
        model: GPT model to use for completions
        embedding_model: Model to use for embeddings
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (seconds)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the OpenAI handler.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, loads from environment
            base_url (str, optional): Custom base URL (e.g., for FastRouter). If None, loads from environment
            model (str): GPT model name
            embedding_model (str): Embedding model name
            max_retries (int): Maximum retry attempts for failed requests
            retry_delay (int): Initial delay between retries in seconds
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Get base URL from parameter or environment (optional)
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL')
        
        # Initialize OpenAI client with optional base_url
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"Using custom base URL: {self.base_url}")
        else:
            self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"OpenAI Handler initialized with model={model}, embedding_model={embedding_model}")
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    raise
                
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a text string.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding vector (list of floats)
            
        Examples:
            >>> handler = OpenAIHandler()
            >>> embedding = handler.get_embedding("Sample text")
            >>> len(embedding)
            1536
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text provided for embedding")
            return []
        
        # Truncate text if too long (max 8191 tokens for ada-002)
        if len(text) > 8000:
            logger.warning(f"Text too long ({len(text)} chars), truncating...")
            text = text[:8000]
        
        def _get_embedding():
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        
        try:
            embedding = self._retry_with_backoff(_get_embedding)
            logger.info(f"Generated embedding for text of length {len(text)}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        
        if not valid_texts:
            logger.warning("No valid texts provided for batch embedding")
            return []
        
        # Truncate long texts
        truncated_texts = [t[:8000] if len(t) > 8000 else t for t in valid_texts]
        
        def _get_embeddings():
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=truncated_texts
            )
            return [item.embedding for item in response.data]
        
        try:
            embeddings = self._retry_with_backoff(_get_embeddings)
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise
    
    def generate_answer(
        self,
        question: str,
        context: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate an answer to a question using GPT with provided context.
        
        Args:
            question (str): User's question
            context (str): Context from retrieved documents
            system_message (str, optional): System message for the model
            temperature (float): Sampling temperature (0.0 to 1.0)
            max_tokens (int): Maximum tokens in response
            
        Returns:
            str: Generated answer
            
        Examples:
            >>> handler = OpenAIHandler()
            >>> context = "The NIST CSF has five functions: Identify, Protect, Detect, Respond, Recover."
            >>> question = "What are the NIST CSF functions?"
            >>> answer = handler.generate_answer(question, context)
        """
        if not system_message:
            system_message = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "If the answer cannot be found in the context, say so. "
                "Always cite the source when possible."
            )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        def _generate():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        try:
            answer = self._retry_with_backoff(_generate)
            logger.info(f"Generated answer for question: {question[:50]}...")
            return answer
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            raise

    def generate_answer_stream(
        self,
        question: str,
        context: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Generate a streaming answer to a question using GPT with provided context.
        Yields tokens as they are generated.
        """
        if not system_message:
            system_message = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "If the answer cannot be found in the context, say so. "
                "Always cite the source when possible."
            )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                        
        except Exception as e:
            logger.error(f"Failed to generate streaming answer: {str(e)}")
            yield f"\n\n[Error generating response: {str(e)}]"
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def test_connection(self) -> bool:
        """
        Test the OpenAI API connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Try to generate a simple embedding
            test_embedding = self.get_embedding("test")
            if test_embedding and len(test_embedding) > 0:
                logger.info("✓ OpenAI API connection successful")
                return True
            return False
        except Exception as e:
            logger.error(f"✗ OpenAI API connection failed: {str(e)}")
            return False


def main():
    """
    Example usage of the OpenAIHandler class.
    """
    print("=" * 60)
    print("OpenAI Handler Module - Examples")
    print("=" * 60)
    
    try:
        # Initialize handler
        handler = OpenAIHandler()
        
        # Example 1: Test connection
        print("\nExample 1: Testing API connection")
        print("-" * 60)
        
        if handler.test_connection():
            print("✓ Successfully connected to OpenAI API")
        else:
            print("✗ Failed to connect to OpenAI API")
            return
        
        # Example 2: Generate single embedding
        print("\n\nExample 2: Generating single embedding")
        print("-" * 60)
        
        sample_text = "The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk."
        embedding = handler.get_embedding(sample_text)
        
        print(f"Text: {sample_text}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # Example 3: Generate batch embeddings
        print("\n\nExample 3: Generating batch embeddings")
        print("-" * 60)
        
        texts = [
            "The Identify function helps organizations understand cybersecurity risks.",
            "The Protect function outlines safeguards to ensure delivery of services.",
            "The Detect function defines activities to identify cybersecurity events."
        ]
        
        embeddings = handler.get_embeddings_batch(texts)
        
        print(f"Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            print(f"  Embedding {i + 1}: dimension {len(emb)}")
        
        # Example 4: Generate answer
        print("\n\nExample 4: Generating answer with GPT")
        print("-" * 60)
        
        context = """
        The NIST Cybersecurity Framework consists of five core functions:
        1. Identify: Understand cybersecurity risks to systems, people, assets, data, and capabilities
        2. Protect: Develop and implement safeguards to ensure delivery of services
        3. Detect: Develop and implement activities to identify cybersecurity events
        4. Respond: Develop and implement activities to take action regarding detected cybersecurity incidents
        5. Recover: Develop and implement activities to maintain resilience and restore capabilities
        """
        
        question = "What are the five functions of the NIST Cybersecurity Framework?"
        
        answer = handler.generate_answer(question, context, temperature=0.3)
        
        print(f"Question: {question}")
        print(f"\nAnswer: {answer}")
        
        # Example 5: Token counting
        print("\n\nExample 5: Token counting")
        print("-" * 60)
        
        long_text = "This is a sample text. " * 100
        token_count = handler.count_tokens(long_text)
        
        print(f"Text length: {len(long_text)} characters")
        print(f"Estimated tokens: {token_count}")
        
    except ValueError as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nMake sure to:")
        print("1. Create a .env file in the project root")
        print("2. Add your OpenAI API key: OPENAI_API_KEY=sk-your-key-here")
        print("3. Or set the OPENAI_API_KEY environment variable")
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
