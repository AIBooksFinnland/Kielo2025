"""
Test script for the Grammar Discriminator
Tests the discriminator with real API calls and sample data
"""

import asyncio
import json
from discriminator import GrammarDiscriminator


def load_api_key():
    """Load API key from users.txt file."""
    try:
        with open("users.txt", "r", encoding="utf-8") as f:
            content = f.read()
            # Find AIBooks1 API key
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "AIBooks1" in line:
                    # API key should be on the line after password
                    for j in range(i, min(i + 3, len(lines))):
                        if "API-Key:" in lines[j]:
                            api_key = lines[j].split(":", 1)[1].strip()
                            # Remove quotes if present
                            if api_key.startswith('"') and api_key.endswith('"'):
                                api_key = api_key[1:-1]
                            return api_key
        return None
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None


async def test_discriminator_real():
    """Test discriminator with real API calls."""
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("❌ No API key found. Cannot test with real API calls.")
        return
    
    print("🔑 API key loaded successfully")
    
    # Initialize discriminator
    discriminator = GrammarDiscriminator(api_key)
    
    # Test corrections - mix of good and problematic ones
    test_corrections = [
        {
            "original_sentence": "Minä olen opiskelut suomea monta vuotta.",
            "explanation": "Partisiippimuodon korjaus: 'opiskelut' pitäisi olla 'opiskellut'.",
            "corrected_sentence": "Minä olen opiskellut suomea monta vuotta."
        },
        {
            "original_sentence": "Tämä on hyvä kirja.",
            "explanation": "Muuta 'hyvä' sanaksi 'erinomainen' paremman ilmaisun vuoksi.",
            "corrected_sentence": "Tämä on erinomainen kirja."
        },
        {
            "original_sentence": "Google Could Platform tarjoaa pilvipalveluita.",
            "explanation": "Kirjoitusvirhe: 'Could' pitäisi olla 'Cloud'.",
            "corrected_sentence": "Google Cloud Platform tarjoaa pilvipalveluita."
        },
        {
            "original_sentence": "Kissa istuu tuolissa.",
            "explanation": "Vaihda 'istuu' sanaksi 'makaa' ilman erityistä syytä.",
            "corrected_sentence": "Kissa makaa tuolissa."
        },
        {
            "original_sentence": "Maailman laajuinen pandemia vaikutti kaikkiin.",
            "explanation": "Yhdyssanavirhe: 'maailman laajuinen' tulisi olla 'maailmanlaajuinen'.",
            "corrected_sentence": "Maailmanlaajuinen pandemia vaikutti kaikkiin."
        }
    ]
    
    original_text = """
    Minä olen opiskelut suomea monta vuotta. Tämä on hyvä kirja.
    Google Could Platform tarjoaa pilvipalveluita. Kissa istuu tuolissa.
    Maailman laajuinen pandemia vaikutti kaikkiin.
    """
    
    print(f"📝 Testing with {len(test_corrections)} corrections...")
    
    try:
        # Test the discriminator
        filtered_corrections, metadata = await discriminator.filter_corrections(
            test_corrections, 
            original_text.strip()
        )
        
        print("\n=== DISCRIMINATOR RESULTS ===")
        print(f"✅ Original corrections: {metadata['original_count']}")
        print(f"✅ Valid corrections: {metadata['filtered_count']}")
        print(f"❌ Rejected corrections: {metadata['rejected_count']}")
        print(f"📊 Quality score: {metadata['quality_score']}/100")
        print(f"📝 Summary: {metadata['summary']}")
        
        print("\n=== VALID CORRECTIONS ===")
        for i, correction in enumerate(filtered_corrections, 1):
            print(f"{i}. {correction['original_sentence']}")
            print(f"   → {correction['corrected_sentence']}")
            print(f"   📋 {correction['explanation']}")
            print()
        
        if 'rejected_reasons' in metadata and metadata['rejected_reasons']:
            print("=== REJECTED CORRECTIONS ===")
            for rejection in metadata['rejected_reasons']:
                print(f"❌ {rejection}")
                print()
        
        # Test quality threshold
        print("=== QUALITY THRESHOLD TEST ===")
        high_threshold_result, high_threshold_meta = await discriminator.filter_corrections(
            test_corrections, 
            original_text.strip(),
            min_quality_score=80
        )
        
        print(f"With 80% quality threshold: {len(high_threshold_result)} corrections passed")
        print(f"Quality score: {high_threshold_meta['quality_score']}/100")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing discriminator: {e}")
        return False


async def test_batch_validation():
    """Test batch validation functionality."""
    
    api_key = load_api_key()
    if not api_key:
        print("❌ No API key found for batch test.")
        return
    
    discriminator = GrammarDiscriminator(api_key)
    
    # Create multiple batches
    batch1 = [
        {
            "original_sentence": "Minä olen opiskelut suomea.",
            "explanation": "Partisiippimuoto virheellinen.",
            "corrected_sentence": "Minä olen opiskellut suomea."
        }
    ]
    
    batch2 = [
        {
            "original_sentence": "Google Could Platform on hyvä.",
            "explanation": "Could → Cloud",
            "corrected_sentence": "Google Cloud Platform on hyvä."
        }
    ]
    
    try:
        print("\n=== BATCH VALIDATION TEST ===")
        results = await discriminator.batch_validate([batch1, batch2])
        
        for i, (corrections, metadata) in enumerate(results, 1):
            print(f"Batch {i}: {len(corrections)} valid corrections")
            print(f"Quality: {metadata.get('quality_score', 'N/A')}/100")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch validation error: {e}")
        return False


async def main():
    """Main test function."""
    print("=== FINNISH GRAMMAR DISCRIMINATOR TESTING ===\n")
    
    # Test discriminator functionality
    success1 = await test_discriminator_real()
    success2 = await test_batch_validation()
    
    if success1 and success2:
        print("\n🎉 All discriminator tests passed successfully!")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")


if __name__ == "__main__":
    asyncio.run(main())