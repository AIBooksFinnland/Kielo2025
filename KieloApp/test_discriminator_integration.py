"""
Test script to demonstrate discriminator integration with the API
"""

import asyncio
import json
import requests

def test_api_with_discriminator():
    """Test the API with discriminator enabled"""
    
    # Login first
    login_response = requests.post("http://localhost:5000/login", 
                                 json={"username": "AIBooks1", "password": "Nalle2016"})
    
    if login_response.status_code != 200:
        print("❌ Login failed")
        return
    
    session_token = login_response.json()["session_token"]
    print(f"✅ Logged in successfully")
    
    # Test text with mixed quality corrections
    test_cases = [
        {
            "name": "High Quality Corrections",
            "text": "Minä olen opiskelut suomea. Google Could Platform on hyvä palvelu. Maailman laajuinen pandemia vaikutti kaikkiin.",
            "expected_quality": "high"
        },
        {
            "name": "Mixed Quality Corrections", 
            "text": "Kissa on musta. Hyvä kirja on pöydällä. Auto on nopeaa.",
            "expected_quality": "mixed"
        },
        {
            "name": "Good Text (No Corrections Needed)",
            "text": "Tämä on hyvin kirjoitettu teksti. Kaikki sanat ovat oikeassa muodossa ja lauserakenne on selkeä.",
            "expected_quality": "high"
        }
    ]
    
    print("\n=== DISCRIMINATOR INTEGRATION TEST ===")
    
    for case in test_cases:
        print(f"\n📝 Testing: {case['name']}")
        print(f"Text: {case['text'][:50]}...")
        
        # Make API call
        response = requests.post("http://localhost:5000/process_sections",
                               json={
                                   "session_token": session_token,
                                   "selected_titles": ["Test"],
                                   "text_for_corrections": case['text'],
                                   "n_responses": 1,
                                   "selected_model": "fast"
                               })
        
        if response.status_code != 200:
            print(f"❌ API call failed: {response.status_code}")
            continue
        
        result = response.json()
        
        if result["results"]:
            corrections = result["results"][0]["corrections"]
            suggestion = result["results"][0]["suggestion"]
            
            print(f"✅ Corrections found: {len(corrections)}")
            print(f"📊 Suggestion: {suggestion}")
            
            # Extract discriminator info from suggestion
            if "Discriminator:" in suggestion:
                discriminator_info = suggestion.split("Discriminator:")[1].strip()
                print(f"🤖 Discriminator: {discriminator_info}")
            
            # Show a few corrections
            for i, correction in enumerate(corrections[:2]):
                print(f"   {i+1}. '{correction['original_sentence'][:30]}...' → '{correction['corrected_sentence'][:30]}...'")
        else:
            print("❌ No results returned")
    
    print("\n🎉 Discriminator integration test completed!")

if __name__ == "__main__":
    test_api_with_discriminator()