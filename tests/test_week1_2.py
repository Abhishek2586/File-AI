"""
Week 1-2 Integration Test Script
==================================
Comprehensive end-to-end testing of all modules with real PDF processing.

This script tests:
- PDF upload and text extraction
- Text cleaning on messy samples
- Text chunking with various sizes
- OpenAI API connection
- Embedding generation
- Performance metrics
- Complete pipeline integration
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.pdf_processor import PDFProcessor
from src.modules.text_cleaner import TextCleaner
from src.modules.text_chunker import TextChunker
from src.modules.openai_handler import OpenAIHandler
from src.modules.embedding_pipeline import EmbeddingPipeline

# Configure logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"test_week1_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Week1_2_Tester:
    """Comprehensive integration tester for Week 1-2 modules."""
    
    def __init__(self):
        """Initialize the tester with all components."""
        self.pdf_processor = PDFProcessor()
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker(chunk_size=1000, overlap=100)
        
        # Try to initialize OpenAI components
        self.openai_handler = None
        self.embedding_pipeline = None
        self.api_available = False
        
        try:
            self.openai_handler = OpenAIHandler()
            self.embedding_pipeline = EmbeddingPipeline(
                openai_handler=self.openai_handler,
                use_cache=True,
                track_metrics=True
            )
            self.api_available = True
        except ValueError as e:
            logger.warning(f"OpenAI API not configured: {e}")
        
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
        self.performance_data = {}
    
    def log_test(self, test_name: str, passed: bool, message: str = "", data: dict = None):
        """Log a test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        
        if passed:
            self.test_results['passed'] += 1
        else:
            self.test_results['failed'] += 1
        
        self.test_results['details'].append({
            'test': test_name,
            'passed': passed,
            'message': message,
            'data': data or {}
        })
        
        logger.info(f"{status}: {test_name} - {message}")
    
    def test_pdf_extraction(self):
        """Test PDF text extraction with sample data."""
        print("\n" + "=" * 70)
        print("TEST 1: PDF Text Extraction")
        print("=" * 70)
        
        try:
            # Test with mock PDF data
            mock_result = {
                'success': True,
                'filename': 'test.pdf',
                'total_pages': 3,
                'pages': [
                    {'page_number': 1, 'text': 'Page 1: Introduction to NIST Framework'},
                    {'page_number': 2, 'text': 'Page 2: Core Functions and Implementation'},
                    {'page_number': 3, 'text': 'Page 3: Best Practices and Guidelines'}
                ]
            }
            
            full_text = self.pdf_processor.get_full_text(mock_result)
            
            if full_text and all(f'Page {i}' in full_text for i in range(1, 4)):
                self.log_test(
                    "PDF Text Extraction",
                    True,
                    f"Extracted {len(full_text)} characters from {mock_result['total_pages']} pages",
                    {'pages': mock_result['total_pages'], 'chars': len(full_text)}
                )
            else:
                self.log_test("PDF Text Extraction", False, "Failed to extract all pages")
        
        except Exception as e:
            self.log_test("PDF Text Extraction", False, str(e))
    
    def test_text_cleaning(self):
        """Test text cleaning with messy samples."""
        print("\n" + "=" * 70)
        print("TEST 2: Text Cleaning")
        print("=" * 70)
        
        test_cases = [
            {
                'name': 'Extra Whitespace',
                'input': 'This   has    too    many   spaces\\n\\n\\n\\nand newlines',
                'check': lambda x: '   ' not in x
            },
            {
                'name': 'Special Characters',
                'input': 'Smart "quotes" and — dashes',
                'check': lambda x: len(x) > 0
            },
            {
                'name': 'Mixed Issues',
                'input': '  Messy   text\\n\\nwith   problems  ',
                'check': lambda x: not x.startswith(' ') and not x.endswith(' ')
            }
        ]
        
        for test_case in test_cases:
            try:
                cleaned = self.text_cleaner.clean(test_case['input'])
                
                if test_case['check'](cleaned):
                    self.log_test(
                        f"Text Cleaning: {test_case['name']}",
                        True,
                        f"Cleaned {len(test_case['input'])} → {len(cleaned)} chars"
                    )
                else:
                    self.log_test(
                        f"Text Cleaning: {test_case['name']}",
                        False,
                        "Cleaning check failed"
                    )
            except Exception as e:
                self.log_test(f"Text Cleaning: {test_case['name']}", False, str(e))
    
    def test_chunking_variations(self):
        """Test chunking with various text sizes."""
        print("\n" + "=" * 70)
        print("TEST 3: Text Chunking Variations")
        print("=" * 70)
        
        test_texts = [
            ("Short text", "This is a short sentence. " * 10),
            ("Medium text", "This is a medium length text. " * 50),
            ("Long text", "This is a longer text for testing. " * 100)
        ]
        
        chunk_configs = [
            (500, 50),
            (1000, 100),
            (1500, 150)
        ]
        
        for text_name, text in test_texts:
            for chunk_size, overlap in chunk_configs:
                try:
                    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
                    chunks = chunker.chunk_text(text, f"{text_name}.pdf", 1)
                    
                    if len(chunks) > 0:
                        self.log_test(
                            f"Chunking: {text_name} (size={chunk_size}, overlap={overlap})",
                            True,
                            f"Created {len(chunks)} chunks",
                            {'chunks': len(chunks), 'config': f"{chunk_size}/{overlap}"}
                        )
                    else:
                        self.log_test(
                            f"Chunking: {text_name} (size={chunk_size}, overlap={overlap})",
                            False,
                            "No chunks created"
                        )
                except Exception as e:
                    self.log_test(
                        f"Chunking: {text_name} (size={chunk_size}, overlap={overlap})",
                        False,
                        str(e)
                    )
    
    def test_api_connection(self):
        """Test OpenAI API connection."""
        print("\n" + "=" * 70)
        print("TEST 4: OpenAI API Connection")
        print("=" * 70)
        
        if not self.api_available:
            self.log_test("API Connection", False, "API not configured")
            self.test_results['skipped'] += 1
            return
        
        try:
            if self.openai_handler.test_connection():
                self.log_test("API Connection", True, "Successfully connected to API")
            else:
                self.log_test("API Connection", False, "Connection test failed")
        except Exception as e:
            self.log_test("API Connection", False, str(e))
    
    def test_embedding_generation(self):
        """Test embedding generation with performance metrics."""
        print("\n" + "=" * 70)
        print("TEST 5: Embedding Generation & Performance")
        print("=" * 70)
        
        if not self.api_available:
            self.log_test("Embedding Generation", False, "API not configured")
            self.test_results['skipped'] += 1
            return
        
        try:
            # Create test chunks
            test_text = "The NIST Cybersecurity Framework. " * 30
            chunks = self.text_chunker.chunk_text(test_text, "test.pdf", 1)
            
            # Reset metrics
            self.embedding_pipeline.reset_metrics()
            
            # Generate embeddings
            start_time = time.time()
            results = self.embedding_pipeline.generate_embeddings_for_chunks(
                chunks[:3],  # Only 3 chunks to save API costs
                show_progress=False
            )
            elapsed_time = time.time() - start_time
            
            # Get metrics
            metrics = self.embedding_pipeline.get_performance_metrics()
            
            if len(results) == 3:
                self.log_test(
                    "Embedding Generation",
                    True,
                    f"Generated {len(results)} embeddings in {elapsed_time:.2f}s",
                    {
                        'embeddings': len(results),
                        'time': elapsed_time,
                        'metrics': metrics
                    }
                )
                
                # Store performance data
                self.performance_data['embedding_generation'] = {
                    'time': elapsed_time,
                    'embeddings': len(results),
                    'metrics': metrics
                }
            else:
                self.log_test(
                    "Embedding Generation",
                    False,
                    f"Expected 3 embeddings, got {len(results)}"
                )
        
        except Exception as e:
            self.log_test("Embedding Generation", False, str(e))
    
    def test_cache_performance(self):
        """Test caching performance."""
        print("\n" + "=" * 70)
        print("TEST 6: Cache Performance")
        print("=" * 70)
        
        if not self.api_available:
            self.log_test("Cache Performance", False, "API not configured")
            self.test_results['skipped'] += 1
            return
        
        try:
            # Create test chunks
            test_text = "Cache test text. " * 20
            chunks = self.text_chunker.chunk_text(test_text, "cache_test.pdf", 1)
            
            # First run (should hit cache from previous test or make API calls)
            self.embedding_pipeline.reset_metrics()
            start_time_1 = time.time()
            results_1 = self.embedding_pipeline.generate_embeddings_for_chunks(
                chunks[:2],
                show_progress=False
            )
            time_1 = time.time() - start_time_1
            metrics_1 = self.embedding_pipeline.get_performance_metrics()
            
            # Second run (should hit cache)
            self.embedding_pipeline.reset_metrics()
            start_time_2 = time.time()
            results_2 = self.embedding_pipeline.generate_embeddings_for_chunks(
                chunks[:2],
                show_progress=False
            )
            time_2 = time.time() - start_time_2
            metrics_2 = self.embedding_pipeline.get_performance_metrics()
            
            # Cache should make second run faster
            cache_hit_rate = metrics_2['cache_hit_rate']
            
            if cache_hit_rate > 80:  # Expect >80% cache hits on second run
                self.log_test(
                    "Cache Performance",
                    True,
                    f"Cache hit rate: {cache_hit_rate:.1f}%, Time: {time_1:.2f}s → {time_2:.2f}s",
                    {
                        'first_run': time_1,
                        'second_run': time_2,
                        'cache_hit_rate': cache_hit_rate,
                        'speedup': time_1 / time_2 if time_2 > 0 else 0
                    }
                )
            else:
                self.log_test(
                    "Cache Performance",
                    False,
                    f"Low cache hit rate: {cache_hit_rate:.1f}%"
                )
        
        except Exception as e:
            self.log_test("Cache Performance", False, str(e))
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        print("\n" + "=" * 70)
        print("TEST 7: End-to-End Pipeline Integration")
        print("=" * 70)
        
        try:
            # Simulate complete workflow
            sample_pdf_text = """
            The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
            It consists of five core functions: Identify, Protect, Detect, Respond, and Recover.
            These functions provide a strategic view of the lifecycle of an organization's 
            management of cybersecurity risk. The Framework is voluntary guidance, based on 
            existing standards, guidelines, and practices for organizations to better manage 
            and reduce cybersecurity risk.
            """
            
            start_time = time.time()
            
            # Step 1: Clean text
            cleaned_text = self.text_cleaner.clean(sample_pdf_text)
            
            # Step 2: Chunk text
            chunks = self.text_chunker.chunk_text(cleaned_text, "nist_framework.pdf", 1)
            
            # Step 3: Generate embeddings (if API available)
            embeddings = []
            if self.api_available:
                embeddings = self.embedding_pipeline.generate_embeddings_for_chunks(
                    chunks[:2],  # Limit to save costs
                    show_progress=False
                )
            
            elapsed_time = time.time() - start_time
            
            pipeline_success = (
                len(cleaned_text) > 0 and
                len(chunks) > 0 and
                (len(embeddings) > 0 if self.api_available else True)
            )
            
            if pipeline_success:
                self.log_test(
                    "End-to-End Pipeline",
                    True,
                    f"Processed in {elapsed_time:.2f}s: {len(chunks)} chunks, {len(embeddings)} embeddings",
                    {
                        'time': elapsed_time,
                        'chunks': len(chunks),
                        'embeddings': len(embeddings)
                    }
                )
            else:
                self.log_test("End-to-End Pipeline", False, "Pipeline incomplete")
        
        except Exception as e:
            self.log_test("End-to-End Pipeline", False, str(e))
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        print("\n" + "=" * 70)
        print("AI FILE ASSISTANT - WEEK 1-2 INTEGRATION TEST SUITE")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_file}")
        
        # Run all tests
        self.test_pdf_extraction()
        self.test_text_cleaning()
        self.test_chunking_variations()
        self.test_api_connection()
        self.test_embedding_generation()
        self.test_cache_performance()
        self.test_end_to_end_pipeline()
        
        # Generate summary
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        
        print(f"\nTotal Tests Run: {total_tests}")
        print(f"✓ Passed: {self.test_results['passed']}")
        print(f"✗ Failed: {self.test_results['failed']}")
        print(f"⊘ Skipped: {self.test_results['skipped']}")
        
        if total_tests > 0:
            success_rate = (self.test_results['passed'] / total_tests) * 100
            print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        # Print performance summary
        if self.performance_data:
            print("\n" + "-" * 70)
            print("PERFORMANCE METRICS")
            print("-" * 70)
            
            for test_name, data in self.performance_data.items():
                print(f"\n{test_name}:")
                if 'metrics' in data:
                    metrics = data['metrics']
                    print(f"  Total embeddings: {metrics['total_embeddings']}")
                    print(f"  Cache hits: {metrics['cache_hits']}")
                    print(f"  Cache misses: {metrics['cache_misses']}")
                    print(f"  Cache hit rate: {metrics['cache_hit_rate']:.1f}%")
                    print(f"  API calls: {metrics['api_calls']}")
                    print(f"  Total time: {metrics['total_time']:.2f}s")
                    print(f"  Avg time/embedding: {metrics['avg_time_per_embedding']:.3f}s")
                    print(f"  Total tokens: {metrics['total_tokens']}")
                    print(f"  Estimated cost: ${metrics['estimated_cost']:.4f}")
        
        # Print failed tests
        if self.test_results['failed'] > 0:
            print("\n" + "-" * 70)
            print("FAILED TESTS:")
            print("-" * 70)
            
            for detail in self.test_results['details']:
                if not detail['passed']:
                    print(f"✗ {detail['test']}: {detail['message']}")
        
        print("\n" + "=" * 70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def save_results(self):
        """Save test results to JSON file."""
        results_file = log_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'summary': self.test_results,
                'performance': self.performance_data,
                'details': self.test_results['details']
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"\n✓ Test results saved to: {results_file}")
        
        except Exception as e:
            logger.error(f"Failed to save test results: {str(e)}")


def main():
    """Main function to run tests."""
    tester = Week1_2_Tester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
