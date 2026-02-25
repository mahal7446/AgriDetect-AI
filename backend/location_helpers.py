"""
Helper functions for location-based alert filtering
"""

def extract_district(address):
    """
    Extract district name from full address
    Handles formats like 'Bangalore, Karnataka' or 'District Mysore, Karnataka'
    Returns: district name in lowercase for matching
    """
    if not address:
        return None
    
    # Split by comma and take the first significant part
    parts = [p.strip() for p in address.split(',')]
    if not parts:
        return None
    
    # The first part is usually the district/city
    district = parts[0].lower()
    
    # Remove common prefixes like 'district'
    district = district.replace('district', '').strip()
    
    return district if district else None

def normalize_district_name(district):
    """
    Normalize district names to handle variations
    E.g., 'Bengaluru' and 'Bangalore' should match
    """
    if not district:
        return None
    
    district = district.lower().strip()
    
    # Common variations (can be expanded)
    variations = {
        # Karnataka
        'bengaluru': 'bengaluru',
        'bengaluru urban': 'bengaluru',
        'bengaluru rural': 'bengaluru',
        'bangalore': 'bengaluru',
        'bangalore urban': 'bengaluru',
        'bangalore rural': 'bengaluru',
        'mysuru': 'mysuru',
        'mysore': 'mysuru',
        'belagavi': 'belagavi',
        'belgaum': 'belagavi',
        'hubballi': 'hubballi',
        'hubli': 'hubballi',
        'tumakuru': 'tumakuru',
        'tumkur': 'tumakuru',
        'shivamogga': 'shivamogga',
        'shimoga': 'shivamogga',
        'vijayapura': 'vijayapura',
        'bijapur': 'vijayapura',
        'kalaburagi': 'kalaburagi',
        'gulbarga': 'kalaburagi',
    }
    
    return variations.get(district, district)
