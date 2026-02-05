"""
Fact Checker Module - NER + Wikipedia Verification
Verifies factual claims in news articles by:
1. Extracting named entities (organizations, locations, dates)
2. Cross-referencing with Wikipedia
3. Detecting unrealistic numerical claims
"""

import spacy
import wikipediaapi
import re
from typing import Dict, List, Tuple

class FactChecker:
    def __init__(self):
        """Initialize spaCy and Wikipedia API"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Requires: python -m spacy download en_core_web_sm
            pass
        
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='FakeNewsDetector/1.0 (Educational Project)'
        )
        
        # Suspicious patterns for quick checks
        self.numerical_checks = {
            'distance': (r'(\d+)\s*(km|kilometer|kilometres)', 500),
            'speed': (r'(\d+)\s*(km/h|kmph|mph)', 350),
            'percentage': (r'(\d+)%', 100),
        }
        
        # Scam/fake news indicators
        self.scam_patterns = [
            r'forward\s+this',
            r'share\s+(urgently|immediately|now)',
            r'before\s+it.?s\s+(deleted|removed|too\s+late)',
            r'(whatsapp|facebook|google)\s+will\s+charge',
            r'send\s+to\s+\d+\s+(people|contacts|friends)',
            r'turn\s+(blue|green|red)',
            r'don.?t\s+ignore',
            r'urgent(ly)?.*message',
            r'breaking.*!\s*',
            r'shocking.*!',
            r'click\s+here\s+(before|now)',
        ]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = {'organizations': [], 'locations': [], 'dates': [], 'infrastructure': []}
        
        for ent in doc.ents:
            if ent.label_ == 'ORG': entities['organizations'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']: entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE': entities['dates'].append(ent.text)
            elif ent.label_ == 'FAC': entities['infrastructure'].append(ent.text)
        
        return entities
    
    def check_numerical_claims(self, text: str) -> List[Dict]:
        """Check for unrealistic numerical claims"""
        issues = []
        for check_type, (pattern, threshold) in self.numerical_checks.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = int(match[0]) if isinstance(match, tuple) else int(match)
                if value > threshold:
                    issues.append({'reason': f'Unrealistic {check_type}: {value}'})
        return issues
    
    def verify_on_wikipedia(self, entity: str) -> Tuple[bool, str]:
        """Verify if entity exists on Wikipedia"""
        page = self.wiki.page(entity)
        if page.exists():
            return True, f"'{entity}' verified on Wikipedia"
        return False, f"Could not verify '{entity}'"

    def analyze(self, text: str) -> Dict:
        """Perform complete fact-checking analysis"""
        entities = self.extract_entities(text)
        numerical_issues = self.check_numerical_claims(text)
        
        verification_results = []
        if entities['organizations']:
            ver_res = self.verify_on_wikipedia(entities['organizations'][0])
            verification_results.append(ver_res)
            
        return {
            'entities': entities,
            'numerical_issues': numerical_issues,
            'verification': verification_results,
            'warnings': [issue['reason'] for issue in numerical_issues]
        }
