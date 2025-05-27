import os
import time
from io import BytesIO

import requests
from PIL import Image


class TopazAPIUpscalerPipeline:
    """
    High quality upscaling using Topaz synchronous API.
    """

    def __init__(self, mode: str = 'enhance'):
        self.topaz_api_key = os.getenv('TOPAZ_API_KEY')
        self.topaz_url = 'https://api.topazlabs.com/image/v1/enhance'
        self.output_height = 4096
        self.output_width = 4096
        self.model = 'Standard V2'
        self.output_format = 'png'
        self.max_retries = 5
        self.backoff_base = 2

    def __call__(self, input_image: Image.Image) -> Image.Image:
        image_bytes = BytesIO()
        input_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        headers = {
            'X-API-Key': self.topaz_api_key,
            'accept': f'image/{self.output_format}',
        }

        files = {
            'image': ('input.png', image_bytes, 'image/png')
        }

        data = {
            'model': self.model,
            'output_height': self.output_height,
            'output_width': self.output_width,
            'output_format': self.output_format
        }

        for attempt in range(self.max_retries):
            response = requests.post(self.topaz_url, headers=headers, files=files, data=data)

            if response.status_code == 200:
                return Image.open(BytesIO(response.content))

            elif response.status_code == 429:
                sleep_time = self.backoff_base ** attempt
                time.sleep(sleep_time)
                continue

            else:
                response.raise_for_status()

        raise Exception("Topaz sync upscaling failed after retries.")
