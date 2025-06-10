import asyncio
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, filename='debug.log', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from openai import AsyncOpenAI



class WordGrammarChecker:
    def __init__(self, api_key, system_prompt, max_concurrent_requests=5, n_responses=1,
                 chosen_model="fast"):
        """
        :param chosen_model: "fast" => use gpt-4o, "slow" => use o3-mini + reasoning_effort=high
        """
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.n_responses = n_responses
        self.text_data = ""

        self.chosen_model = chosen_model  # "fast" or "slow"

        # Ensure the API key doesn't have quotes around it (which might be in users.txt)
        if self.api_key and (self.api_key.startswith('"') or self.api_key.startswith("'")):
            self.api_key = self.api_key.strip('"\'')
            
        self.client = AsyncOpenAI(api_key=self.api_key)

    def create_payload(self):
        # Decide model
        if self.chosen_model == "slow":
            model_name = "o3-mini"
            reasoning_effort = True
            system_role_key = "developer"
        else:
            model_name = "gpt-4o"
            reasoning_effort = False
            system_role_key = "system"

        payload = {
            "model": model_name,
            "messages": [
                {"role": system_role_key, "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Tämä on teksti, josta ehkä voit löytää virheitä, "
                        "mutta älä kuitenkaan väkisin yritä löytää virheitä "
                        "sieltä, missä niitä ei ole:\n\n" + self.text_data
                    )
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "my_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the schema"
                            },
                            "type": {
                                "type": "string",
                                "enum": ["object"]
                            },
                            "properties": {
                                "type": "object",
                                "properties": {
                                    "corrections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "original_sentence": {
                                                    "type": ["string","null"]
                                                },
                                                "explanation": {
                                                    "type": ["string","null"]
                                                },
                                                "corrected_sentence": {
                                                    "type": ["string","null"]
                                                }
                                            },
                                            "required": [
                                                "original_sentence",
                                                "explanation",
                                                "corrected_sentence"
                                            ],
                                            "additionalProperties": False
                                        }
                                    },
                                    "suggestion": {
                                        "type": "string"
                                    }
                                },
                                "required": ["corrections","suggestion"],
                                "additionalProperties": False
                            }
                        },
                        "$defs": {},
                        "required": ["name","type","properties"],
                        "additionalProperties": False
                    }
                }
            }
        }
        # If "slow", we set reasoning_effort="high"
        if reasoning_effort:
            payload["reasoning_effort"] = "high"

        return payload

    async def make_api_call(self, payload):
        async with self.semaphore:
            try:
                logging.debug(f"Making OpenAI API call with model: {payload.get('model')}")
                response = await self.client.chat.completions.create(**payload)
                logging.debug(f"Received response: {response}")
                return response
            except Exception as e:
                logging.error(f"OpenAI call failed: {e}")
                return None

    def extract_corrections(self, response_model):
        if not response_model:
            return [], ""
        resp_dict = response_model.model_dump()
        choices = resp_dict.get("choices", [])
        if not choices:
            return [], ""

        content = choices[0].get("message",{}).get("content","")
        try:
            data = json.loads(content)
            props = data.get("properties", {})
            corrections = props.get("corrections", [])
            suggestion = props.get("suggestion", "")
            return corrections, suggestion
        except json.JSONDecodeError:
            return [], ""

    async def process_text(self, text):
        self.text_data = text
        payload = self.create_payload()

        logging.info(f"API Key Status: {'*****VALID*****' if self.api_key else 'MISSING OR EMPTY'}")
        logging.info(f"API Key First 10 chars: {self.api_key[:10] if self.api_key else 'NONE'}...")
        logging.info(f"Model being used: {payload.get('model', 'unknown')}")

        # For debugging - to be commented out when API is working
        # mock_response = {
        #     "corrections": [
        #         {
        #             "original_sentence": "Poliisi on hyödyntää tekoälyä rikoksien selvittämisessä.",
        #             "explanation": "Virheessä on ylimääräinen apuverbi 'on'. Kun käytetään verbiä perusmuodossa, ei tarvita apuverbiä.",
        #             "corrected_sentence": "Poliisi hyödyntää tekoälyä rikoksien selvittämisessä."
        #         }
        #     ],
        #     "suggestion": "Tekstissä oli kielioppivirhe, jossa käytettiin ylimääräistä apuverbiä 'on' yhdessä perusmuotoisen verbin kanssa."
        # }

        calls = [self.make_api_call(payload) for _ in range(self.n_responses)]
        responses = await asyncio.gather(*calls)

        responses = [r for r in responses if r is not None]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open("word_grammar_checker.log", "a", encoding="utf-8") as f:
            f.write(f"--- NEW SET OF RESPONSES at {timestamp} ---\n")
            if responses:
                for idx, resp_model in enumerate(responses):
                    resp_dict = resp_model.model_dump()
                    f.write(f"Response {idx+1}:\n")
                    f.write(json.dumps(resp_dict, ensure_ascii=False, indent=2))
                    f.write("\n\n")
            else:
                f.write("No valid responses received.\n\n")
                f.write("API KEY ISSUE: Using mock response for testing\n")

        if not responses:
            logging.error("No valid responses received.")
            return [], responses

        results_per_response = []
        for resp_model in responses:
            corrections, suggestion = self.extract_corrections(resp_model)
            results_per_response.append((corrections, suggestion))

        return results_per_response, responses
