"""
Text Chunker Module
===================
Splits large documents into smaller, manageable chunks for embedding.

This module handles:
- Intelligent text chunking with configurable size
- Sentence boundary preservation
- Chunk overlap for context preservation
- Metadata tracking for each chunk
"""

import re
import logging
from typing import List, Dict
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    Data class representing a text chunk with metadata.
    
    Attributes:
        text (str): The chunk text content
        metadata (dict): Metadata about the chunk including:
            - source_file: Original filename
            - page_number: Page number (if applicable)
            - chunk_index: Index of this chunk
            - start_char: Starting character position
            - end_char: Ending character position
    """
    text: str
    metadata: Dict[str, any]
    
    def __repr__(self):
        return f"TextChunk(length={len(self.text)}, metadata={self.metadata})"


class TextChunker:
    """
    A class to split text into chunks while preserving context.
    
    Attributes:
        chunk_size (int): Target size for each chunk in characters
        overlap (int): Number of overlapping characters between chunks
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size (int): Target size for each chunk (default: 1000 characters)
            overlap (int): Overlap between chunks (default: 100 characters)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
        
        logger.info(f"TextChunker initialized with chunk_size={chunk_size}, overlap={overlap}")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        # Split on sentence boundaries (., !, ?) followed by space and capital letter
        # or end of string
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_text(
        self, 
        text: str, 
        source_file: str = "unknown",
        page_number: int = None
    ) -> List[TextChunk]:
        """
        Split text into chunks with overlap, preserving sentence boundaries.
        
        Args:
            text (str): Text to chunk
            source_file (str): Name of the source file
            page_number (int): Page number if applicable
            
        Returns:
            list: List of TextChunk objects
            
        Examples:
            >>> chunker = TextChunker(chunk_size=100, overlap=20)
            >>> text = "This is a sample text. " * 50
            >>> chunks = chunker.chunk_text(text, "sample.pdf", 1)
            >>> len(chunks) > 1
            True
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text provided")
            return []
        
        logger.info(f"Chunking text of length {len(text)} from {source_file}")
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk = TextChunk(
                    text=current_chunk.strip(),
                    metadata={
                        'source_file': source_file,
                        'page_number': page_number,
                        'chunk_index': chunk_index,
                        'start_char': current_start,
                        'end_char': current_start + len(current_chunk)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                # Get the last overlap characters from current chunk
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text) - len(sentence) - 1
                chunk_index += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk = TextChunk(
                text=current_chunk.strip(),
                metadata={
                    'source_file': source_file,
                    'page_number': page_number,
                    'chunk_index': chunk_index,
                    'start_char': current_start,
                    'end_char': current_start + len(current_chunk)
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        
        return chunks
    
    def chunk_document(
        self,
        pages: List[Dict[str, any]],
        source_file: str = "unknown"
    ) -> List[TextChunk]:
        """
        Chunk an entire document that has multiple pages.
        
        Args:
            pages (list): List of page dictionaries with 'page_number' and 'text' keys
            source_file (str): Name of the source file
            
        Returns:
            list: List of TextChunk objects from all pages
            
        Examples:
            >>> chunker = TextChunker()
            >>> pages = [
            ...     {'page_number': 1, 'text': 'Page 1 text...'},
            ...     {'page_number': 2, 'text': 'Page 2 text...'}
            ... ]
            >>> chunks = chunker.chunk_document(pages, "document.pdf")
        """
        all_chunks = []
        
        for page in pages:
            page_number = page.get('page_number', None)
            text = page.get('text', '')
            
            if text:
                chunks = self.chunk_text(text, source_file, page_number)
                all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created from {source_file}: {len(all_chunks)}")
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, any]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks (list): List of TextChunk objects
            
        Returns:
            dict: Statistics including count, avg length, min/max length
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0
            }
        
        lengths = [len(chunk.text) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths)
        }


def main():
    """
    Example usage of the TextChunker class.
    """
    print("=" * 60)
    print("Text Chunker Module - Examples")
    print("=" * 60)
    
    # Initialize chunker
    chunker = TextChunker(chunk_size=200, overlap=50)
    
    # Example 1: Simple text chunking
    print("\nExample 1: Simple text chunking")
    print("-" * 60)
    
    sample_text = """
    The NIST Cybersecurity Framework provides a policy framework of computer security 
    guidance for how private sector organizations can assess and improve their ability 
    to prevent, detect, and respond to cyber attacks. The Framework consists of standards, 
    guidelines, and best practices to manage cybersecurity-related risk. It was created 
    through collaboration between government and the private sector.
    
    The Framework is organized around five key functions: Identify, Protect, Detect, 
    Respond, and Recover. These functions provide a high-level, strategic view of the 
    lifecycle of an organization's management of cybersecurity risk.
    """
    
    chunks = chunker.chunk_text(sample_text, "NIST_Framework.pdf", 1)
    
    print(f"Original text length: {len(sample_text)}")
    print(f"Number of chunks created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Length: {len(chunk.text)}")
        print(f"  Metadata: {chunk.metadata}")
        print(f"  Text preview: {chunk.text[:100]}...")
    
    # Example 2: Document with multiple pages
    print("\n\n" + "=" * 60)
    print("Example 2: Multi-page document chunking")
    print("=" * 60)
    
    pages = [
        {
            'page_number': 1,
            'text': "This is page 1. " * 50
        },
        {
            'page_number': 2,
            'text': "This is page 2. " * 50
        }
    ]
    
    all_chunks = chunker.chunk_document(pages, "multi_page.pdf")
    
    print(f"\nTotal chunks from all pages: {len(all_chunks)}")
    
    # Group by page
    page_groups = {}
    for chunk in all_chunks:
        page = chunk.metadata['page_number']
        if page not in page_groups:
            page_groups[page] = []
        page_groups[page].append(chunk)
    
    for page, chunks in page_groups.items():
        print(f"  Page {page}: {len(chunks)} chunks")
    
    # Example 3: Chunk statistics
    print("\n\n" + "=" * 60)
    print("Example 3: Chunk statistics")
    print("=" * 60)
    
    stats = chunker.get_chunk_statistics(all_chunks)
    
    print(f"\nStatistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Average length: {stats['avg_length']:.2f} characters")
    print(f"  Min length: {stats['min_length']} characters")
    print(f"  Max length: {stats['max_length']} characters")
    
    # Example 4: Demonstrating overlap
    print("\n\n" + "=" * 60)
    print("Example 4: Demonstrating chunk overlap")
    print("=" * 60)
    
    short_text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    overlap_chunks = chunker.chunk_text(short_text, "overlap_demo.txt")
    
    print(f"\nChunk size: {chunker.chunk_size}, Overlap: {chunker.overlap}")
    
    for i, chunk in enumerate(overlap_chunks):
        print(f"\nChunk {i + 1}: {chunk.text}")
        if i > 0:
            # Show overlap with previous chunk
            prev_chunk = overlap_chunks[i - 1]
            overlap_text = prev_chunk.text[-chunker.overlap:]
            if overlap_text in chunk.text:
                print(f"  ↳ Overlaps with previous: '{overlap_text[:30]}...'")


if __name__ == "__main__":
    main()
