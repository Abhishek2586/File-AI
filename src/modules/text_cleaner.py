"""
Text Cleaner Module
===================
Cleans and preprocesses extracted text from PDFs.

This module handles:
- Whitespace normalization
- Special character removal
- OCR error correction
- Header/footer removal
- Text encoding normalization
"""

import re
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    A class to clean and preprocess text extracted from PDFs.
    
    Attributes:
        preserve_formatting (bool): Whether to preserve bullet points and lists
    """
    
    def __init__(self, preserve_formatting: bool = True):
        """
        Initialize the text cleaner.
        
        Args:
            preserve_formatting (bool): If True, preserves bullet points and numbered lists
        """
        self.preserve_formatting = preserve_formatting
    
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove unnecessary whitespace while preserving paragraph structure.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove spaces at the beginning and end of lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        return text.strip()
    
    def fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with OCR errors corrected
        """
        # Common OCR substitutions
        ocr_fixes = {
            r'\bl\b': 'I',  # lowercase L to uppercase I when standalone
            r'\bO\b': '0',  # uppercase O to zero when standalone in numbers
            r'rn': 'm',     # rn often misread as m
            r'\|': 'l',     # pipe to lowercase L
            r'~': '-',      # tilde to hyphen
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def remove_special_characters(self, text: str) -> str:
        """
        Remove or normalize special characters.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with special characters removed/normalized
        """
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)
        
        return text
    
    def detect_and_remove_headers_footers(self, text: str) -> str:
        """
        Attempt to detect and remove common headers and footers.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with headers/footers removed
        """
        lines = text.split('\n')
        
        if len(lines) < 3:
            return text
        
        # Remove lines that are likely page numbers
        cleaned_lines = []
        for line in lines:
            # Skip lines that are just page numbers
            if re.match(r'^\s*\d+\s*$', line):
                continue
            # Skip lines that are just "Page X" or "X of Y"
            if re.match(r'^\s*(Page\s+)?\d+(\s+of\s+\d+)?\s*$', line, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def preserve_list_formatting(self, text: str) -> str:
        """
        Ensure bullet points and numbered lists are preserved.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with preserved list formatting
        """
        if not self.preserve_formatting:
            return text
        
        # Ensure bullet points have proper spacing
        text = re.sub(r'([•·∙▪▫])', r'\n\1 ', text)
        
        # Ensure numbered lists have proper spacing
        text = re.sub(r'\n(\d+\.)', r'\n\1 ', text)
        
        return text
    
    def normalize_encoding(self, text: str) -> str:
        """
        Normalize text encoding to handle various character encodings.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized encoding
        """
        # Replace common encoding issues
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def clean(self, text: str) -> str:
        """
        Apply all cleaning operations to the text.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
            
        Examples:
            >>> cleaner = TextCleaner()
            >>> dirty_text = "This   has    extra   spaces\\n\\n\\n\\nand newlines"
            >>> clean_text = cleaner.clean(dirty_text)
            >>> print(clean_text)
            'This has extra spaces\\n\\nand newlines'
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text provided")
            return ""
        
        logger.info(f"Cleaning text of length {len(text)}")
        
        # Apply cleaning operations in sequence
        text = self.normalize_encoding(text)
        text = self.remove_special_characters(text)
        text = self.fix_common_ocr_errors(text)
        text = self.detect_and_remove_headers_footers(text)
        text = self.preserve_list_formatting(text)
        text = self.remove_extra_whitespace(text)
        
        logger.info(f"Cleaned text length: {len(text)}")
        
        return text


def main():
    """
    Example usage of the TextCleaner class.
    """
    print("=" * 60)
    print("Text Cleaner Module - Examples")
    print("=" * 60)
    
    # Initialize cleaner
    cleaner = TextCleaner(preserve_formatting=True)
    
    # Example 1: Remove extra whitespace
    print("\nExample 1: Removing extra whitespace")
    print("-" * 60)
    dirty_text = "This   has    too    many   spaces\n\n\n\n\nand newlines"
    clean_text = cleaner.remove_extra_whitespace(dirty_text)
    print(f"Before: {repr(dirty_text)}")
    print(f"After:  {repr(clean_text)}")
    
    # Example 2: Fix OCR errors
    print("\n\nExample 2: Fixing OCR errors")
    print("-" * 60)
    ocr_text = "The nurnber is l23 and the word is rn"
    fixed_text = cleaner.fix_common_ocr_errors(ocr_text)
    print(f"Before: {ocr_text}")
    print(f"After:  {fixed_text}")
    
    # Example 3: Remove special characters
    print("\n\nExample 3: Removing special characters")
    print("-" * 60)
    special_text = 'Smart quotes and apostrophes with dashes'
    normalized_text = cleaner.remove_special_characters(special_text)
    print(f"Before: {special_text}")
    print(f"After:  {normalized_text}")
    
    # Example 4: Complete cleaning
    print("\n\nExample 4: Complete text cleaning")
    print("-" * 60)
    messy_text = """
    This   is   a   messy   document
    
    
    
    • First bullet point
    • Second bullet point
    
    1. First numbered item
    2. Second numbered item
    
    Page 5
    
    Some more text with quotes and dashes.
    
    42
    """
    
    cleaned = cleaner.clean(messy_text)
    print("Before:")
    print(messy_text)
    print("\nAfter:")
    print(cleaned)
    
    # Example 5: Preserve formatting
    print("\n\nExample 5: List formatting preservation")
    print("-" * 60)
    list_text = "Items:•Item 1•Item 2•Item 3"
    formatted = cleaner.preserve_list_formatting(list_text)
    print(f"Before: {list_text}")
    print(f"After:\n{formatted}")


if __name__ == "__main__":
    main()
