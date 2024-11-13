import os
import tempfile
import streamlit as st
from PIL import Image
import ollama
import time
from transformers import MarianMTModel, MarianTokenizer
import re
from diffusers import StableDiffusionPipeline
import torch
import io
import pygame
from gtts import gTTS

# Model configurations
LLAVA_MODEL = "llava:13b"
OLLAMA_MODEL = "llama3.1:8b"

# Translation languages and models
LANGUAGE_MODELS = {
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Portuguese": "Helsinki-NLP/opus-mt-tc-big-en-pt"
}

languages_dict = {
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Portuguese": "pt",
    "English": "en"
}

def extract_tags_with_llava(image):
    """
    Use LLaVA to describe the image and extract 5 tags.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image_path = temp_file.name
            image.save(temp_file, format="PNG")

        image = Image.open(image_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        res = ollama.chat(
            model=LLAVA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe the image, then add in a separate paragraph five suggested English single words as tags (separated by commas).',
                    'images': [image_path]
                }
            ],
        )
        content = res['message']['content']
        tags = content.splitlines()[-1]

        if os.path.exists(image_path):
            os.remove(image_path)

        return tags
    except Exception as e:
        st.error(f"Error during LLaVA tag extraction: {e}")
        return None

def generate_story_with_ollama(tags, genre, length):
    word_count = 300 if length == "short" else 600 if length == "medium" else 1000
    prompt = (f"Write a {genre} story using the following elements: {tags}. "
              f"The story should be around {word_count} words.")
    try:
        messages = [{'role': 'user', 'content': prompt}]
        stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)

        story = ''
        for chunk in stream:
            story_chunk = chunk['message']['content']
            story += story_chunk

        return story
    except Exception as e:
        st.error(f"Error during story generation: {e}")
        return None

def word_count(story):
    words = story.split()
    return len(words)

def translation(language, story, model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    paragraphs = story.split("\n\n")
    translated_paragraphs = []

    for paragraph in paragraphs:
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            encoded_length = len(tokenizer.encode(current_chunk + sentence, return_tensors='pt')[0])
            if encoded_length < 512:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())

        translated_chunks = []
        for chunk in chunks:
            inputs = tokenizer.prepare_seq2seq_batch([chunk], return_tensors='pt')
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            translated_chunks.append(translated_text)

        translated_paragraph = ' '.join(translated_chunks)
        translated_paragraphs.append(translated_paragraph)

    translated_story = "\n\n".join(translated_paragraphs)
    return translated_story

def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        tts.save(temp_file.name)
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    os.remove(temp_file.name)

def generate_image_from_text(prompt):
    """Generate an image from a text prompt using Stable Diffusion."""
    device = torch.device("mps")  # Set device to MPS for Apple Silicon
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    image = pipe(prompt).images[0]

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()  # Return the byte content


def extract_tags_with_llava_image_generation(image_bytes):
    """Use LLaVA to describe the uploaded image and extract 5 tags."""
    try:
        # Save the image temporarily to a specific path on the desktop
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=os.path.expanduser("~/Desktop")) as temp_file:
            image_path = temp_file.name
            with open(image_path, "wb") as f:
                f.write(image_bytes)
        
        # Load and display the saved image
        image = Image.open(image_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Start the timer
        start = time.time()

        # Use LLaVA to describe the image and extract tags
        res = ollama.chat(
            model=LLAVA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe the image, then add in a separate paragraph five suggested English single words as tags (separated by commas).',
                    'images': [image_path]
                }
            ]
        )

        # Check if there is a response from the model
        if 'message' not in res or 'content' not in res['message']:
            st.error("Failed to retrieve a valid response from the model.")
            return None

        # Extract the tags from the response
        content = res['message']['content']
        tags = content.splitlines()[-1]  # Assume the last line contains the tags

        # Clean up: Delete the temporary file after processing
        if os.path.exists(image_path):
            os.remove(image_path)
        
        return tags
    except Exception as e:
        st.error(f"Error during LLaVA tag extraction: {e}")
        return None

def uploading_image_main():
    st.title("AI-Generated Story from Image (Upload)")

    finished = False

    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None
    if "tags" not in st.session_state:
        st.session_state.tags = []
    if "story" not in st.session_state:
        st.session_state.story = ""
    if "translated_story" not in st.session_state:
        st.session_state.translated_story = ""

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        tags = extract_tags_with_llava(image)

        if tags:
            genre = st.selectbox("Choose a story genre", ["fantasy", "mystery", "romance", "sci-fi", "adventure"])
            length = st.selectbox("Choose the length of the story", ["short", "medium", "long"])

            if st.button("Generate Story"):
                st.session_state.story = generate_story_with_ollama(tags, genre, length)

            if st.session_state.story:
                st.subheader("Generated Story")
                st.write(st.session_state.story)
                count = word_count(st.session_state.story)

                translate_option = st.checkbox("Translate Story")
                if translate_option:
                    language = st.selectbox("Choose a language", list(LANGUAGE_MODELS.keys()))
                    if st.button("Translate"):
                        model_name = LANGUAGE_MODELS[language]
                        st.session_state.translated_story = translation(language, st.session_state.story, model_name)

                if st.session_state.translated_story:
                    st.subheader(f"Translated Story ({language})")
                    st.write(st.session_state.translated_story)

                    if st.button("Play Translated Story Audio"):
                        text_to_speech(st.session_state.translated_story, languages_dict[language])
                        finished = True

                if st.button("Play Original Story Audio"):
                    text_to_speech(st.session_state.story, "en")
                    finished = True


                if finished:
                    st.subheader("General Evaluation")
                    st.text_area("Provide feedback on the story")
                    if st.button("Send Evaluation"):
                        st.write("Thank you for your evaluation!")





def image_generation_main():
    st.title("AI-Generated Story from Image (Generate)")

    finished = False

    if "image_bytes_gen" not in st.session_state:
        st.session_state.image_bytes_gen = None
    if "tags_gen" not in st.session_state:
        st.session_state.tags_gen = []
    if "story_gen" not in st.session_state:
        st.session_state.story_gen = ""
    if "translated_story_gen" not in st.session_state:
        st.session_state.translated_story_gen = ""

    prompt = st.text_input("Enter a prompt to generate an image")
    if st.button("Generate Image"):
        if prompt:
            st.session_state.image_bytes_gen = generate_image_from_text(prompt)
        else:
            st.warning("Please enter a prompt to generate an image.")

    if st.session_state.image_bytes_gen:
        st.session_state.tags_gen = extract_tags_with_llava_image_generation(st.session_state.image_bytes_gen)

        if st.session_state.tags_gen:
            genre = st.selectbox("Choose a story genre", ["fantasy", "mystery", "romance", "sci-fi", "adventure"])
            length = st.selectbox("Choose the length of the story", ["short", "medium", "long"])

            if st.button("Generate Story"):
                st.session_state.story_gen = generate_story_with_ollama(st.session_state.tags_gen, genre, length)

            if st.session_state.story_gen:
                st.subheader("Generated Story")
                st.write(st.session_state.story_gen)

                translate_option = st.checkbox("Translate Story")
                if translate_option:
                    language = st.selectbox("Choose a language", list(LANGUAGE_MODELS.keys()))
                    if st.button("Translate"):
                        model_name = LANGUAGE_MODELS[language]
                        st.session_state.translated_story_gen = translation(language, st.session_state.story_gen, model_name)

                if st.session_state.translated_story_gen:
                    st.subheader(f"Translated Story ({language})")
                    st.write(st.session_state.translated_story_gen)

                    if st.button("Play Translated Story Audio"):
                        text_to_speech(st.session_state.translated_story_gen, languages_dict[language])
                        finished = True

                if st.button("Play Original Story Audio"):
                    text_to_speech(st.session_state.story_gen, "en")
                    finished = True


                if finished:
                    st.subheader("General Evaluation")
                    st.text_area("Provide feedback on the story")
                    if st.button("Send Evaluation"):
                        st.write("Thank you for your evaluation!")
                        

def main():
    page = st.sidebar.selectbox("Choose between Generating or Uploading an Image", ["Upload Image", "Generate Image"])

    if page == "Upload Image":
        uploading_image_main()
    elif page == "Generate Image":
        image_generation_main()

if __name__ == "__main__":
    main()
