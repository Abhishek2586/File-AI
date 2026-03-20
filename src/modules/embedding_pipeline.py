"""
Embedding Pipeline Module
==========================
Orchestrates the process of generating embeddings for text chunks.

This module handles:
- Batch processing of text chunks
- Progress tracking
- Caching to avoid re-computation
- Integration with OpenAI handler
"""

import os
import json
import logging
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Import local modules
from src.modules.openai_handler import OpenAIHandler
from src.modules.text_chunker import TextChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    A class to manage the embedding generation pipeline.
    
    Attributes:
        openai_handler: OpenAI handler instance
        cache_dir: Directory to store cached embeddings
        use_cache: Whether to use caching
    """
    
    def __init__(
        self,
        openai_handler: OpenAIHandler = None,
        cache_dir: str = "./embeddings_cache",
        use_cache: bool = True,
        track_metrics: bool = True
    ):
        """
        Initialize the embedding pipeline.
        
        Args:
            openai_handler: OpenAI handler instance (creates new one if None)
            cache_dir: Directory for caching embeddings
            use_cache: Whether to enable caching
            track_metrics: Whether to track performance metrics
        """
        self.openai_handler = openai_handler or OpenAIHandler()
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.track_metrics = track_metrics
        
        # Performance metrics
        self.metrics = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'total_time': 0.0,
            'total_tokens': 0,
            'estimated_cost': 0.0
        }
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache enabled at {self.cache_dir}")
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text string.
        
        Args:
            text (str): Text to generate key for
            
        Returns:
            str: Cache key (hash of the text)
        """
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> List[float]:
        """
        Load embedding from cache.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            list: Embedding vector or None if not found
        """
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data['embedding']
            except Exception as e:
                logger.warning(f"Failed to load from cache: {str(e)}")
                return None
        
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]):
        """
        Save embedding to cache.
        
        Args:
            cache_key (str): Cache key
            embedding (list): Embedding vector
        """
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({'embedding': embedding}, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {str(e)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text, using cache if available.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding vector
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_embedding = self._load_from_cache(cache_key)
        
        if cached_embedding is not None:
            if self.track_metrics:
                self.metrics['cache_hits'] += 1
                self.metrics['total_embeddings'] += 1
            logger.debug(f"Loaded embedding from cache")
            return cached_embedding
        
        # Generate new embedding
        embedding = self.openai_handler.get_embedding(text)
        
        # Update metrics
        if self.track_metrics:
            self.metrics['cache_misses'] += 1
            self.metrics['api_calls'] += 1
            self.metrics['total_embeddings'] += 1
            self.metrics['total_time'] += time.time() - start_time
            
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            tokens = len(text) // 4
            self.metrics['total_tokens'] += tokens
            
            # Estimate cost (text-embedding-ada-002: $0.0001 per 1K tokens)
            self.metrics['estimated_cost'] += (tokens / 1000) * 0.0001
        
        # Save to cache
        self._save_to_cache(cache_key, embedding)
        
        return embedding
    
    def generate_embeddings_for_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 10,
        show_progress: bool = True
    ) -> List[Tuple[TextChunk, List[float]]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks (list): List of TextChunk objects
            batch_size (int): Number of chunks to process in each batch
            show_progress (bool): Whether to show progress bar
            
        Returns:
            list: List of tuples (chunk, embedding)
            
        Examples:
            >>> pipeline = EmbeddingPipeline()
            >>> chunks = [TextChunk("Sample text", {"source": "test.pdf"})]
            >>> results = pipeline.generate_embeddings_for_chunks(chunks)
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        results = []
        
        # Process chunks with progress bar
        iterator = tqdm(chunks, desc="Generating embeddings") if show_progress else chunks
        
        for chunk in iterator:
            try:
                embedding = self.generate_embedding(chunk.text)
                results.append((chunk, embedding))
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {chunk.metadata.get('chunk_index')}: {str(e)}")
                # Continue with next chunk instead of failing completely
                continue
        
        logger.info(f"Successfully generated {len(results)} embeddings")
        
        return results
    
    def process_document(
        self,
        chunks: List[TextChunk],
        batch_size: int = 10
    ) -> Dict[str, any]:
        """
        Process a complete document and generate embeddings.
        
        Args:
            chunks (list): List of TextChunk objects from a document
            batch_size (int): Batch size for processing
            
        Returns:
            dict: Processing results including:
                - total_chunks: Number of chunks processed
                - successful: Number of successful embeddings
                - failed: Number of failed embeddings
                - embeddings: List of (chunk, embedding) tuples
        """
        logger.info(f"Processing document with {len(chunks)} chunks")
        
        embeddings = self.generate_embeddings_for_chunks(chunks, batch_size)
        
        result = {
            'total_chunks': len(chunks),
            'successful': len(embeddings),
            'failed': len(chunks) - len(embeddings),
            'embeddings': embeddings
        }
        
        logger.info(
            f"Document processing complete: "
            f"{result['successful']} successful, {result['failed']} failed"
        )
        
        return result
    
    def clear_cache(self):
        """
        Clear the embedding cache.
        """
        if not self.use_cache:
            logger.warning("Caching is disabled")
            return
        
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get statistics about the cache.
        
        Returns:
            dict: Cache statistics
        """
        if not self.use_cache or not self.cache_dir.exists():
            return {'enabled': False, 'files': 0, 'size_mb': 0}
        
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'enabled': True,
            'files': len(cache_files),
            'size_mb': total_size / (1024 * 1024)
        }
    
    def get_performance_metrics(self) -> Dict[str, any]:
        """
        Get performance metrics for the pipeline.
        
        Returns:
            dict: Performance metrics including:
                - total_embeddings: Total embeddings generated
                - cache_hits: Number of cache hits
                - cache_misses: Number of cache misses
                - cache_hit_rate: Percentage of cache hits
                - api_calls: Number of API calls made
                - total_time: Total processing time in seconds
                - avg_time_per_embedding: Average time per embedding
                - total_tokens: Estimated total tokens processed
                - estimated_cost: Estimated API cost in USD
        """
        metrics = self.metrics.copy()
        
        # Calculate derived metrics
        if metrics['total_embeddings'] > 0:
            metrics['cache_hit_rate'] = (metrics['cache_hits'] / metrics['total_embeddings']) * 100
            metrics['avg_time_per_embedding'] = metrics['total_time'] / metrics['total_embeddings']
        else:
            metrics['cache_hit_rate'] = 0.0
            metrics['avg_time_per_embedding'] = 0.0
        
        return metrics
    
    def reset_metrics(self):
        """
        Reset performance metrics.
        """
        self.metrics = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'total_time': 0.0,
            'total_tokens': 0,
            'estimated_cost': 0.0
        }
        logger.info("Performance metrics reset")


def main():
    """
    Example usage of the EmbeddingPipeline class.
    """
    print("=" * 60)
    print("Embedding Pipeline Module - Examples")
    print("=" * 60)
    
    try:
        from src.modules.text_chunker import TextChunker
        
        # Initialize pipeline
        pipeline = EmbeddingPipeline(use_cache=True)
        
        # Example 1: Test OpenAI connection
        print("\nExample 1: Testing OpenAI connection")
        print("-" * 60)
        
        if pipeline.openai_handler.test_connection():
            print("✓ Successfully connected to OpenAI API")
        else:
            print("✗ Failed to connect to OpenAI API")
            return
        
        # Example 2: Generate embedding for single text
        print("\n\nExample 2: Generating single embedding")
        print("-" * 60)
        
        sample_text = "The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk."
        embedding = pipeline.generate_embedding(sample_text)
        
        print(f"Text: {sample_text}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # Example 3: Generate embeddings for chunks
        print("\n\nExample 3: Generating embeddings for chunks")
        print("-" * 60)
        
        # Create sample chunks
        chunker = TextChunker(chunk_size=200, overlap=50)
        
        sample_document = """
        The NIST Cybersecurity Framework provides a policy framework of computer security 
        guidance for how private sector organizations can assess and improve their ability 
        to prevent, detect, and respond to cyber attacks. The Framework consists of standards, 
        guidelines, and best practices to manage cybersecurity-related risk.
        
        The Framework is organized around five key functions: Identify, Protect, Detect, 
        Respond, and Recover. These functions provide a high-level, strategic view of the 
        lifecycle of an organization's management of cybersecurity risk.
        """
        
        chunks = chunker.chunk_text(sample_document, "NIST_Framework.pdf", 1)
        
        print(f"Created {len(chunks)} chunks from sample document")
        
        # Generate embeddings
        results = pipeline.generate_embeddings_for_chunks(chunks, show_progress=True)
        
        print(f"\nGenerated {len(results)} embeddings")
        for i, (chunk, embedding) in enumerate(results[:2]):  # Show first 2
            print(f"\nChunk {i + 1}:")
            print(f"  Text: {chunk.text[:80]}...")
            print(f"  Embedding dimension: {len(embedding)}")
        
        # Example 4: Cache statistics
        print("\n\nExample 4: Cache statistics")
        print("-" * 60)
        
        cache_stats = pipeline.get_cache_stats()
        
        print(f"Cache enabled: {cache_stats['enabled']}")
        print(f"Cached files: {cache_stats['files']}")
        print(f"Cache size: {cache_stats['size_mb']:.2f} MB")
        
        # Example 5: Process complete document
        print("\n\nExample 5: Processing complete document")
        print("-" * 60)
        
        result = pipeline.process_document(chunks)
        
        print(f"\nProcessing Results:")
        print(f"  Total chunks: {result['total_chunks']}")
        print(f"  Successful: {result['successful']}")
        print(f"  Failed: {result['failed']}")
        print(f"  Success rate: {(result['successful'] / result['total_chunks'] * 100):.1f}%")
        
    except ImportError as e:
        print(f"\n✗ Import error: {str(e)}")
        print("Make sure all required modules are in the modules/ directory")
    except ValueError as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nMake sure to:")
        print("1. Create a .env file in the project root")
        print("2. Add your OpenAI API key: OPENAI_API_KEY=sk-your-key-here")
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
