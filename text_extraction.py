import ollama
from loguru import logger



if __name__ == "__main__":
    image_path = "/Users/michael/Desktop/Bildschirmfoto 2024-07-20 um 20.51.42.png"

    message = {
        'role': 'user',
        'content': 'Describe this image:',
        'images': [image_path]
    }

    response = ollama.chat(
        model="llava:13b",
        messages=[message]
    )

    logger.info("Response received")

    logger.info(response["message"]["content"])