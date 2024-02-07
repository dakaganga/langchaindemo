from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import requests
import os

class CustomLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            api_url = 'https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud/'
            API_TOKEN=os.environ.get("API_TOKEN")
            print(API_TOKEN)
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 1024,
                "temprature":0.6,
                "top_p":0.9}
                
            }

            headers={"Authorization":f"Bearer {API_TOKEN}"}
            response = requests.post(api_url, json=payload, headers=headers, verify=False)
            response.raise_for_status()
            #print("API Response:", response.json())

            return response.json()[0]['generated_text'] 
