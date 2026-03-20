"""
Week 1 Integration Test Script
===============================
Tests all Week 1 modules together in an end-to-end workflow.

This script tests:
- PDF text extraction
- Text cleaning
- Text chunking
- OpenAI API connection
- Embedding generation
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.pdf_processor import PDFProcessor
from src.modules.text_cleaner import TextCleaner
from src.modules.text_chunker import TextChunker
from src.modules.openai_handler import OpenAIHandler
from src.modules.embedding_pipeline import EmbeddingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Week1Tester:
    """
    A class to run comprehensive tests for Week 1 modules.
    """
    
    def __init__(self):
        """Initialize the tester with all required components."""
        self.pdf_processor = PDFProcessor()
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker(chunk_size=1000, overlap=100)
        self.openai_handler = None  # Will initialize if API key available
        self.embedding_pipeline = None
        
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
    
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """
        Log a test result.
        
        Args:
            test_name (str): Name of the test
            passed (bool): Whether the test passed
            message (str): Additional message
        """
        status = "✓ PASS" if passed else "✗ FAIL"
        
        if passed:
            self.test_results['passed'] += 1
        else:
            self.test_results['failed'] += 1
        
        self.test_results['details'].append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
        
        logger.info(f"{status}: {test_name} - {message}")
    
    def test_pdf_processor(self):
        """Test PDF processing functionality."""
        print("\n" + "=" * 60)
        print("TEST 1: PDF Processor")
        print("=" * 60)
        
        # Test with a sample text file (since we may not have PDFs)
        test_text = "Sample PDF content for testing."
        
        try:
            # Test that the processor can be initialized
            processor = PDFProcessor()
            self.log_test("PDF Processor Initialization", True, "Processor created successfully")
            
            # Test get_full_text method
            mock_result = {
                'success': True,
                'pages': [
                    {'page_number': 1, 'text': 'Page 1 text'},
                    {'page_number': 2, 'text': 'Page 2 text'}
                ]
            }
            full_text = processor.get_full_text(mock_result)
            
            if full_text and 'Page 1' in full_text and 'Page 2' in full_text:
                self.log_test("PDF Full Text Extraction", True, f"Extracted {len(full_text)} characters")
            else:
                self.log_test("PDF Full Text Extraction", False, "Failed to combine pages")
                
        except Exception as e:
            self.log_test("PDF Processor", False, str(e))
    
    def test_text_cleaner(self):
        """Test text cleaning functionality."""
        print("\n" + "=" * 60)
        print("TEST 2: Text Cleaner")
        print("=" * 60)
        
        try:
            cleaner = TextCleaner()
            
            # Test 1: Whitespace removal
            dirty_text = "This   has    extra   spaces\n\n\n\nand newlines"
            clean_text = cleaner.remove_extra_whitespace(dirty_text)
            
            if "   " not in clean_text:
                self.log_test("Whitespace Removal", True, "Extra spaces removed")
            else:
                self.log_test("Whitespace Removal", False, "Extra spaces still present")
            
            # Test 2: Special character removal
            special_text = 'Smart "quotes" and — dashes'
            normalized = cleaner.remove_special_characters(special_text)
            
            if '"' in normalized or '"' in normalized:
                self.log_test("Special Character Removal", False, "Smart quotes not normalized")
            else:
                self.log_test("Special Character Removal", True, "Special characters normalized")
            
            # Test 3: Complete cleaning
            messy_text = "This   is   messy\n\n\n\ntext"
            cleaned = cleaner.clean(messy_text)
            
            if cleaned and len(cleaned) > 0:
                self.log_test("Complete Text Cleaning", True, f"Cleaned {len(messy_text)} -> {len(cleaned)} chars")
            else:
                self.log_test("Complete Text Cleaning", False, "Cleaning failed")
                
        except Exception as e:
            self.log_test("Text Cleaner", False, str(e))
    
    def test_text_chunker(self):
        """Test text chunking functionality."""
        print("\n" + "=" * 60)
        print("TEST 3: Text Chunker")
        print("=" * 60)
        
        try:
            chunker = TextChunker(chunk_size=200, overlap=50)
            
            # Test 1: Basic chunking
            sample_text = "This is a sentence. " * 50
            chunks = chunker.chunk_text(sample_text, "test.pdf", 1)
            
            if len(chunks) > 1:
                self.log_test("Text Chunking", True, f"Created {len(chunks)} chunks")
            else:
                self.log_test("Text Chunking", False, "Failed to create multiple chunks")
            
            # Test 2: Chunk metadata
            if chunks and chunks[0].metadata:
                metadata = chunks[0].metadata
                has_required_fields = all(
                    key in metadata 
                    for key in ['source_file', 'chunk_index', 'start_char', 'end_char']
                )
                
                if has_required_fields:
                    self.log_test("Chunk Metadata", True, "All required fields present")
                else:
                    self.log_test("Chunk Metadata", False, "Missing metadata fields")
            
            # Test 3: Chunk overlap
            if len(chunks) >= 2:
                chunk1_end = chunks[0].text[-50:]
                chunk2_start = chunks[1].text[:50]
                
                # Check if there's some overlap
                has_overlap = any(word in chunk2_start for word in chunk1_end.split()[-5:])
                
                if has_overlap:
                    self.log_test("Chunk Overlap", True, "Overlap detected between chunks")
                else:
                    self.log_test("Chunk Overlap", False, "No overlap detected")
            
            # Test 4: Statistics
            stats = chunker.get_chunk_statistics(chunks)
            
            if stats['total_chunks'] == len(chunks):
                self.log_test("Chunk Statistics", True, f"Stats: {stats}")
            else:
                self.log_test("Chunk Statistics", False, "Statistics mismatch")
                
        except Exception as e:
            self.log_test("Text Chunker", False, str(e))
    
    def test_openai_handler(self):
        """Test OpenAI API integration."""
        print("\n" + "=" * 60)
        print("TEST 4: OpenAI Handler")
        print("=" * 60)
        
        try:
            # Try to initialize OpenAI handler
            self.openai_handler = OpenAIHandler()
            self.log_test("OpenAI Handler Initialization", True, "Handler created")
            
            # Test API connection
            if self.openai_handler.test_connection():
                self.log_test("OpenAI API Connection", True, "Successfully connected to API")
                
                # Test embedding generation
                try:
                    test_text = "This is a test for embedding generation."
                    embedding = self.openai_handler.get_embedding(test_text)
                    
                    if embedding and len(embedding) == 1536:
                        self.log_test("Embedding Generation", True, f"Generated {len(embedding)}-dim embedding")
                    else:
                        self.log_test("Embedding Generation", False, f"Unexpected embedding dimension: {len(embedding)}")
                        
                except Exception as e:
                    self.log_test("Embedding Generation", False, str(e))
            else:
                self.log_test("OpenAI API Connection", False, "Failed to connect")
                
        except ValueError as e:
            self.log_test("OpenAI Handler", False, "API key not found - Set OPENAI_API_KEY in .env")
            self.test_results['skipped'] += 1
            logger.warning("Skipping OpenAI tests - no API key configured")
        except Exception as e:
            self.log_test("OpenAI Handler", False, str(e))
    
    def test_embedding_pipeline(self):
        """Test embedding pipeline."""
        print("\n" + "=" * 60)
        print("TEST 5: Embedding Pipeline")
        print("=" * 60)
        
        if not self.openai_handler:
            self.log_test("Embedding Pipeline", False, "Skipped - OpenAI handler not available")
            self.test_results['skipped'] += 1
            return
        
        try:
            # Initialize pipeline
            self.embedding_pipeline = EmbeddingPipeline(
                openai_handler=self.openai_handler,
                use_cache=True
            )
            
            self.log_test("Embedding Pipeline Initialization", True, "Pipeline created")
            
            # Create sample chunks
            sample_text = "The NIST Cybersecurity Framework. " * 10
            chunks = self.text_chunker.chunk_text(sample_text, "test.pdf", 1)
            
            # Generate embeddings
            results = self.embedding_pipeline.generate_embeddings_for_chunks(
                chunks[:2],  # Only process 2 chunks to save API calls
                show_progress=False
            )
            
            if len(results) == 2:
                self.log_test("Batch Embedding Generation", True, f"Generated {len(results)} embeddings")
            else:
                self.log_test("Batch Embedding Generation", False, f"Expected 2, got {len(results)}")
            
            # Test cache
            cache_stats = self.embedding_pipeline.get_cache_stats()
            
            if cache_stats['enabled'] and cache_stats['files'] > 0:
                self.log_test("Embedding Cache", True, f"{cache_stats['files']} files cached")
            else:
                self.log_test("Embedding Cache", False, "Cache not working properly")
                
        except Exception as e:
            self.log_test("Embedding Pipeline", False, str(e))
    
    def test_integration(self):
        """Test complete integration of all modules."""
        print("\n" + "=" * 60)
        print("TEST 6: End-to-End Integration")
        print("=" * 60)
        
        try:
            # Simulate complete workflow
            sample_text = """
            The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
            It consists of five core functions: Identify, Protect, Detect, Respond, and Recover.
            These functions provide a strategic view of cybersecurity risk management.
            """
            
            # Step 1: Clean text
            cleaned_text = self.text_cleaner.clean(sample_text)
            
            # Step 2: Chunk text
            chunks = self.text_chunker.chunk_text(cleaned_text, "integration_test.pdf", 1)
            
            if len(chunks) > 0:
                self.log_test("Integration: Clean + Chunk", True, f"Created {len(chunks)} chunks")
            else:
                self.log_test("Integration: Clean + Chunk", False, "No chunks created")
                return
            
            # Step 3: Generate embeddings (if OpenAI available)
            if self.embedding_pipeline:
                try:
                    results = self.embedding_pipeline.generate_embeddings_for_chunks(
                        chunks[:1],  # Only 1 chunk to save API calls
                        show_progress=False
                    )
                    
                    if len(results) == 1:
                        self.log_test("Integration: Full Pipeline", True, "Complete workflow successful")
                    else:
                        self.log_test("Integration: Full Pipeline", False, "Embedding generation failed")
                        
                except Exception as e:
                    self.log_test("Integration: Full Pipeline", False, str(e))
            else:
                self.log_test("Integration: Full Pipeline", False, "Skipped - OpenAI not configured")
                self.test_results['skipped'] += 1
                
        except Exception as e:
            self.log_test("End-to-End Integration", False, str(e))
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        print("\n" + "=" * 60)
        print("AI FILE ASSISTANT - WEEK 1 TEST SUITE")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tests
        self.test_pdf_processor()
        self.test_text_cleaner()
        self.test_text_chunker()
        self.test_openai_handler()
        self.test_embedding_pipeline()
        self.test_integration()
        
        # Generate summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        
        print(f"\nTotal Tests Run: {total_tests}")
        print(f"✓ Passed: {self.test_results['passed']}")
        print(f"✗ Failed: {self.test_results['failed']}")
        print(f"⊘ Skipped: {self.test_results['skipped']}")
        
        if total_tests > 0:
            success_rate = (self.test_results['passed'] / total_tests) * 100
            print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        # Print failed tests
        if self.test_results['failed'] > 0:
            print("\n" + "-" * 60)
            print("FAILED TESTS:")
            print("-" * 60)
            
            for detail in self.test_results['details']:
                if not detail['passed']:
                    print(f"✗ {detail['test']}: {detail['message']}")
        
        print("\n" + "=" * 60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Save results to file
        self.save_results()
    
    def save_results(self):
        """Save test results to a log file."""
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"week1_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        try:
            with open(log_file, 'w') as f:
                f.write("AI FILE ASSISTANT - WEEK 1 TEST RESULTS\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Total Tests: {self.test_results['passed'] + self.test_results['failed']}\n")
                f.write(f"Passed: {self.test_results['passed']}\n")
                f.write(f"Failed: {self.test_results['failed']}\n")
                f.write(f"Skipped: {self.test_results['skipped']}\n\n")
                
                f.write("DETAILED RESULTS:\n")
                f.write("-" * 60 + "\n")
                
                for detail in self.test_results['details']:
                    status = "PASS" if detail['passed'] else "FAIL"
                    f.write(f"[{status}] {detail['test']}: {detail['message']}\n")
            
            print(f"\n✓ Test results saved to: {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {str(e)}")


def main():
    """Main function to run tests."""
    tester = Week1Tester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
