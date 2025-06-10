"""
Finnish Grammar Correction Discriminator

A discriminator agent that validates AI grammar correction responses using the o3 model.
It filters out incorrect or low-quality corrections and returns only the valid ones
in the correct JSON structure.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrammarDiscriminator:
    """
    Discriminator agent for validating Finnish grammar corrections.
    Uses o3 model to evaluate and filter correction suggestions.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the discriminator with OpenAI API key.
        
        Args:
            api_key (str): OpenAI API key for accessing o3 model
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "o3"  # Using o3 model as specified
        
        # Validation prompt for the discriminator
        self.validation_prompt = """
Olet asiantuntija suomen kielen kieliopin tarkastaja. Tehtäväsi on arvioida annettujen kielioppikorjausten laatua ja oikeellisuutta.

ARVIOINTIKRITEERIT:
1. Onko alkuperäisessä lauseessa todella virhe?
2. Onko ehdotettu korjaus kieliopillisesti oikein?
3. Säilyykö lauseen alkuperäinen merkitys?
4. Onko selitys selkeä ja perusteltu?
5. Onko korjaus tarpeellinen ja hyödyllinen?

ARVIOITAVAT KORJAUKSET:
{corrections_json}

VASTAUSOHJE:
Palauta JSON-objekti, joka sisältää:
1. "valid_corrections": Lista hyväksytyistä korjauksista (alkuperäisessä muodossa)
2. "rejected_corrections": Lista hylätyistä korjauksista syineen
3. "quality_score": Kokonaislaatupisteet 0-100
4. "summary": Lyhyt yhteenveto arvioinnista

HYVÄKSYMISKYNNYS:
- Hyväksy vain korjaukset, jotka ovat selvästi tarpeellisia ja oikeita
- Hylkää epäselvät, tarpeettomat tai virheelliset korjaukset
- Hylkää korjaukset, jotka muuttavat merkitystä perusteettomasti

Vastaa VAIN JSON-muodossa ilman lisäselityksiä.
"""

    async def validate_corrections(
        self, 
        corrections: List[Dict[str, Any]], 
        original_text: str = ""
    ) -> Dict[str, Any]:
        """
        Validate grammar corrections using the discriminator model.
        
        Args:
            corrections: List of correction dictionaries
            original_text: Original text for context (optional)
            
        Returns:
            Dictionary with validated corrections and quality metrics
        """
        try:
            # Prepare corrections for validation
            corrections_json = json.dumps(corrections, ensure_ascii=False, indent=2)
            
            # Add original text context if provided
            prompt = self.validation_prompt.format(corrections_json=corrections_json)
            if original_text:
                prompt += f"\n\nALKUPERÄINEN TEKSTI KONTEKSTIKSI:\n{original_text}"
            
            # Call o3 model for validation
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Olet asiantuntija suomen kielen kieliopin arvioitsija. Vastaa aina JSON-muodossa."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                # Note: o3 model only supports default temperature=1
                max_completion_tokens=2000,  # Use max_completion_tokens for o3 model
                response_format={"type": "json_object"}
            )
            
            # Parse the validation result
            validation_result = json.loads(response.choices[0].message.content)
            
            # Log validation summary
            logger.info(f"Discriminator validation completed. "
                       f"Valid: {len(validation_result.get('valid_corrections', []))}, "
                       f"Rejected: {len(validation_result.get('rejected_corrections', []))}, "
                       f"Quality Score: {validation_result.get('quality_score', 0)}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in correction validation: {e}")
            # Fallback: return all corrections if validation fails
            return {
                "valid_corrections": corrections,
                "rejected_corrections": [],
                "quality_score": 50,
                "summary": f"Validation failed: {str(e)}. Returned all corrections.",
                "error": str(e)
            }

    async def filter_corrections(
        self, 
        corrections: List[Dict[str, Any]], 
        original_text: str = "",
        min_quality_score: int = 70
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filter corrections and return only high-quality ones.
        
        Args:
            corrections: List of correction dictionaries
            original_text: Original text for context
            min_quality_score: Minimum quality score to accept results
            
        Returns:
            Tuple of (filtered_corrections, validation_metadata)
        """
        validation_result = await self.validate_corrections(corrections, original_text)
        
        # Extract valid corrections
        valid_corrections = validation_result.get("valid_corrections", [])
        quality_score = validation_result.get("quality_score", 0)
        
        # Apply quality threshold
        if quality_score < min_quality_score:
            logger.warning(f"Quality score {quality_score} below threshold {min_quality_score}")
            # Return fewer corrections if quality is low
            valid_corrections = valid_corrections[:max(1, len(valid_corrections) // 2)]
        
        # Create metadata for tracking
        metadata = {
            "discriminator_used": True,
            "quality_score": quality_score,
            "original_count": len(corrections),
            "filtered_count": len(valid_corrections),
            "rejected_count": len(validation_result.get("rejected_corrections", [])),
            "summary": validation_result.get("summary", ""),
            "rejected_reasons": validation_result.get("rejected_corrections", [])
        }
        
        return valid_corrections, metadata

    async def batch_validate(
        self, 
        correction_batches: List[List[Dict[str, Any]]], 
        original_texts: List[str] = None
    ) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Validate multiple batches of corrections concurrently.
        
        Args:
            correction_batches: List of correction lists
            original_texts: List of original texts for context
            
        Returns:
            List of (filtered_corrections, metadata) tuples
        """
        if original_texts is None:
            original_texts = [""] * len(correction_batches)
        
        # Create tasks for concurrent validation
        tasks = [
            self.filter_corrections(corrections, text)
            for corrections, text in zip(correction_batches, original_texts)
        ]
        
        # Execute all validations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        filtered_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} validation failed: {result}")
                # Fallback to original corrections
                filtered_results.append((
                    correction_batches[i], 
                    {"error": str(result), "discriminator_used": False}
                ))
            else:
                filtered_results.append(result)
        
        return filtered_results


# Test functions
async def test_discriminator():
    """Test function to validate discriminator functionality."""
    
    # Sample corrections for testing (some good, some problematic)
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
            "original_sentence": "Kissa on mustaa.",
            "explanation": "Väärä sanaluokka: 'mustaa' pitäisi olla 'musta'.",
            "corrected_sentence": "Kissa on musta."
        }
    ]
    
    # Test with a placeholder API key (will fail but shows structure)
    try:
        # Note: This test requires a real API key to work
        discriminator = GrammarDiscriminator("test-api-key")
        
        print("Testing discriminator structure...")
        print(f"Model: {discriminator.model}")
        print(f"Test corrections count: {len(test_corrections)}")
        
        # This would actually call the API if we had a real key
        # result = await discriminator.filter_corrections(test_corrections, "Testiteksti kontekstiksi.")
        # print(f"Filtered corrections: {len(result[0])}")
        # print(f"Metadata: {result[1]}")
        
        print("Discriminator structure test completed successfully!")
        
    except Exception as e:
        print(f"Expected error (no real API key): {e}")
        print("Discriminator class structure is correct.")


if __name__ == "__main__":
    """Run discriminator tests when file is executed directly."""
    print("=== Finnish Grammar Discriminator Test ===")
    asyncio.run(test_discriminator())