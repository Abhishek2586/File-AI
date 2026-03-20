"""
PDF Processor Module
====================
Extracts text from PDF files using PyMuPDF (fitz).

This module handles:
- Multiple PDF file processing
- Text extraction with metadata
- Error handling for corrupted PDFs
- Page-by-page text extraction
"""

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. PDF processing will not work.")
    print("Install with: pip install PyMuPDF")

from typing import List, Dict, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    A class to handle PDF text extraction operations.
    
    Attributes:
        None
    """
    
    def __init__(self):
        """Initialize the PDF processor."""
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Dictionary containing:
                - filename (str): Name of the PDF file
                - total_pages (int): Total number of pages
                - pages (list): List of dictionaries with page_number and text
                - success (bool): Whether extraction was successful
                - error (str, optional): Error message if extraction failed
                
        Examples:
            >>> processor = PDFProcessor()
            >>> result = processor.extract_text_from_pdf("document.pdf")
            >>> print(result['filename'])
            'document.pdf'
        """
        result = {
            'filename': Path(pdf_path).name,
            'total_pages': 0,
            'pages': [],
            'success': False,
            'error': None
        }
        
        if not PYMUPDF_AVAILABLE:
            result['error'] = "PyMuPDF not installed. Install with: pip install PyMuPDF"
            logger.error(result['error'])
            return result
        
        try:
            # Open the PDF file
            doc = fitz.open(pdf_path)
            result['total_pages'] = len(doc)
            
            logger.info(f"Processing {result['filename']} with {result['total_pages']} pages")
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                result['pages'].append({
                    'page_number': page_num + 1,
                    'text': text
                })
            
            doc.close()
            result['success'] = True
            logger.info(f"Successfully extracted text from {result['filename']}")
            
        except Exception as e:
            error_msg = f"Error processing {pdf_path}: {str(e)}"
            logger.error(error_msg)
            result['error'] = error_msg
            
        return result
    
    def extract_text_from_pdfs(self, pdf_files: List[str]) -> List[Dict[str, any]]:
        """
        Extract text from multiple PDF files.
        
        Args:
            pdf_files (list): List of paths to PDF files
            
        Returns:
            list: List of dictionaries, each containing extraction results for one PDF
            
        Examples:
            >>> processor = PDFProcessor()
            >>> results = processor.extract_text_from_pdfs(["doc1.pdf", "doc2.pdf"])
            >>> print(len(results))
            2
        """
        results = []
        
        logger.info(f"Starting batch processing of {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            result = self.extract_text_from_pdf(pdf_path)
            results.append(result)
        
        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
        
        return results
    
    def get_full_text(self, extraction_result: Dict[str, any]) -> str:
        """
        Combine all pages into a single text string.
        
        Args:
            extraction_result (dict): Result from extract_text_from_pdf
            
        Returns:
            str: Combined text from all pages
            
        Examples:
            >>> processor = PDFProcessor()
            >>> result = processor.extract_text_from_pdf("document.pdf")
            >>> full_text = processor.get_full_text(result)
        """
        if not extraction_result['success']:
            return ""
        
        return "\n\n".join([page['text'] for page in extraction_result['pages']])


def main():
    """
    Example usage of the PDFProcessor class.
    """
    # Initialize processor
    processor = PDFProcessor()
    
    # Example 1: Process a single PDF
    print("=" * 60)
    print("Example 1: Processing a single PDF")
    print("=" * 60)
    
    # Note: Replace with actual PDF path for testing
    sample_pdf = "data/test_data/sample.pdf"
    
    print(f"\nAttempting to process: {sample_pdf}")
    print("(This will fail if the file doesn't exist - that's expected for demo)")
    
    result = processor.extract_text_from_pdf(sample_pdf)
    
    if result['success']:
        print(f"\n✓ Successfully processed: {result['filename']}")
        print(f"  Total pages: {result['total_pages']}")
        print(f"  First 200 characters of page 1:")
        if result['pages']:
            print(f"  {result['pages'][0]['text'][:200]}...")
    else:
        print(f"\n✗ Failed to process: {result['error']}")
    
    # Example 2: Process multiple PDFs
    print("\n" + "=" * 60)
    print("Example 2: Processing multiple PDFs")
    print("=" * 60)
    
    pdf_list = [
        "data/test_data/NIST_CSWP_29.pdf",
        "data/test_data/NIST_IR_8596_iprd.pdf"
    ]
    
    print(f"\nAttempting to process {len(pdf_list)} PDFs")
    results = processor.extract_text_from_pdfs(pdf_list)
    
    print(f"\nProcessing Summary:")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['filename']}: {r['total_pages']} pages")
    
    # Example 3: Get full text
    print("\n" + "=" * 60)
    print("Example 3: Getting full text from a document")
    print("=" * 60)
    
    if results and results[0]['success']:
        full_text = processor.get_full_text(results[0])
        print(f"\nTotal characters extracted: {len(full_text)}")
        print(f"First 300 characters:\n{full_text[:300]}...")


if __name__ == "__main__":
    main()
