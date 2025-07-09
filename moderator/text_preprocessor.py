"""
Text preprocessing utilities for text moderation.
"""

import re
import string
from typing import List


class TextPreprocessor:
    """Handles text cleaning and preprocessing for all datasets."""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s]')
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub(' ', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self.email_pattern.sub(' ', text)
    
    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from text."""
        return self.mention_pattern.sub(' ', text)
    
    def remove_hashtags(self, text: str) -> str:
        """Remove hashtags from text."""
        return self.hashtag_pattern.sub(' ', text)
    
    def remove_numbers(self, text: str) -> str:
        """Remove standalone numbers from text."""
        return self.number_pattern.sub(' ', text)
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and spaces."""
        return self.special_chars_pattern.sub(' ', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to single spaces."""
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_extra_spaces(self, text: str) -> str:
        """Remove extra spaces and normalize whitespace."""
        return ' '.join(text.split())
    
    def clean_text(self, text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_mentions: bool = True,
                   remove_hashtags: bool = False,  # Keep hashtags as they might be informative
                   remove_numbers: bool = True,
                   remove_special_chars: bool = True,
                   to_lowercase: bool = True) -> str:
        """
        Apply comprehensive text cleaning.
        
        Args:
            text: Input text to clean
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove hashtags
            remove_numbers: Whether to remove standalone numbers
            remove_special_chars: Whether to remove special characters
            to_lowercase: Whether to convert to lowercase
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Start with the original text
        cleaned = text
        
        # Apply cleaning steps
        if remove_urls:
            cleaned = self.remove_urls(cleaned)
        
        if remove_emails:
            cleaned = self.remove_emails(cleaned)
        
        if remove_mentions:
            cleaned = self.remove_mentions(cleaned)
        
        if remove_hashtags:
            cleaned = self.remove_hashtags(cleaned)
        
        if remove_numbers:
            cleaned = self.remove_numbers(cleaned)
        
        if remove_special_chars:
            cleaned = self.remove_special_chars(cleaned)
        
        # Normalize whitespace
        cleaned = self.normalize_whitespace(cleaned)
        
        if to_lowercase:
            cleaned = self.to_lowercase(cleaned)
        
        # Final cleanup
        cleaned = self.remove_extra_spaces(cleaned)
        
        return cleaned
    
    def preprocess_texts(self, texts: List[str], **kwargs) -> List[str]:
        """
        Preprocess a list of texts.
        
        Args:
            texts: List of texts to preprocess
            **kwargs: Arguments to pass to clean_text method
            
        Returns:
            List of preprocessed texts
        """
        return [self.clean_text(text, **kwargs) for text in texts]
    
    def get_text_stats(self, texts: List[str]) -> dict:
        """
        Get statistics about the texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with text statistics
        """
        if not texts:
            return {}
        
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        stats = {
            'total_texts': len(texts),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_words': sum(word_counts) / len(word_counts),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'empty_texts': sum(1 for text in texts if len(text.strip()) == 0)
        }
        
        return stats


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "Check out this link: https://example.com @user #hashtag",
        "Email me at test@example.com for more info!",
        "This has 123 numbers and special chars: !@#$%",
        "UPPERCASE TEXT with Mixed Case"
    ]
    
    print("Original texts:")
    for i, text in enumerate(test_texts):
        print(f"{i+1}: {text}")
    
    print("\nCleaned texts:")
    cleaned = preprocessor.preprocess_texts(test_texts)
    for i, text in enumerate(cleaned):
        print(f"{i+1}: {text}")
    
    print("\nText statistics:")
    stats = preprocessor.get_text_stats(cleaned)
    for key, value in stats.items():
        print(f"{key}: {value}")
