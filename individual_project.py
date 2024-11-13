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
        #st.image(image, caption="Uploaded Image", use_column_width=True)

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
        #st.image(image, caption="Uploaded Image", use_column_width=True)

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


# modify to see the image just once!
def uploading_image_main():
    st.title("AI-Generated Story from Image (Upload)")

    # Initialize session states
    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None
    if "image_display" not in st.session_state:
        st.session_state.image_display = None
    if "tags" not in st.session_state:
        st.session_state.tags = []
    if "story" not in st.session_state:
        st.session_state.story = ""
    if "translated_story" not in st.session_state:
        st.session_state.translated_story = ""
    if "show_evaluation" not in st.session_state:
        st.session_state.show_evaluation = False  # Track evaluation box visibility
    if "feedback_text" not in st.session_state:
        st.session_state.feedback_text = ""  # Store feedback text

    # Upload image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Only re-extract tags if the image is new
        image_bytes = uploaded_image.getvalue()
        if st.session_state.image_bytes != image_bytes:
            st.session_state.image_bytes = image_bytes
            st.session_state.image_display = Image.open(io.BytesIO(image_bytes))
            tags = extract_tags_with_llava(st.session_state.image_display)
            st.session_state.tags = tags

        # Display the image from session state
        if st.session_state.image_display:
            st.image(st.session_state.image_display, caption="Uploaded Image", use_column_width=True)

        # Story generation based on tags
        if st.session_state.tags:
            genre = st.selectbox("Choose a story genre", ["fantasy", "mystery", "romance", "sci-fi", "adventure"])
            length = st.selectbox("Choose the length of the story", ["short", "medium", "long"])

            if st.button("Generate Story"):
                st.session_state.story = generate_story_with_ollama(st.session_state.tags, genre, length)

            # Display the generated story
            if st.session_state.story:
                st.subheader("Generated Story")
                st.write(st.session_state.story)

                # Option to translate the story
                translate_option = st.checkbox("Translate Story")
                if translate_option:
                    language = st.selectbox("Choose a language", list(LANGUAGE_MODELS.keys()))
                    if st.button("Translate"):
                        model_name = LANGUAGE_MODELS[language]
                        st.session_state.translated_story = translation(language, st.session_state.story, model_name)

                # Display translated story if available
                if st.session_state.translated_story:
                    st.subheader(f"Translated Story ({language})")
                    st.write(st.session_state.translated_story)

                    if st.button("Play Translated Story Audio"):
                        text_to_speech(st.session_state.translated_story, languages_dict[language])
                        st.session_state.show_evaluation = True  # Show evaluation after audio plays

                # Play original story audio
                if st.button("Play Original Story Audio"):
                    text_to_speech(st.session_state.story, "en")
                    st.session_state.show_evaluation = True  # Show evaluation after audio plays

                # Display evaluation section if the evaluation flag is set
                if st.session_state.show_evaluation:
                    st.subheader("General Evaluation")
                    
                    # Display the feedback text area linked to the session state
                    feedback = st.text_area("Provide feedback on the story", value=st.session_state.feedback_text)
                    
                    # When "Send Evaluation" is pressed
                    if st.button("Send Evaluation"):
                        st.write("Thank you for your evaluation!")
                        st.session_state.feedback_text = ""  # Clear feedback text after submission
                        st.session_state.show_evaluation = False  # Optionally hide evaluation after submission


def image_generation_main():
    st.title("AI-Generated Story from Image (Generate)")

    # Initialize session states
    if "image_bytes_gen" not in st.session_state:
        st.session_state.image_bytes_gen = None
    if "tags_gen" not in st.session_state:
        st.session_state.tags_gen = []
    if "story_gen" not in st.session_state:
        st.session_state.story_gen = ""
    if "translated_story_gen" not in st.session_state:
        st.session_state.translated_story_gen = ""
    if "show_evaluation_gen" not in st.session_state:
        st.session_state.show_evaluation_gen = False  # Track evaluation box visibility
    if "feedback_text_gen" not in st.session_state:
        st.session_state.feedback_text_gen = ""  # Store feedback text

    # Prompt for image generation
    prompt = st.text_input("Enter a prompt to generate an image")
    if st.button("Generate Image"):
        if prompt:
            # Generate and display image
            st.session_state.image_bytes_gen = generate_image_from_text(prompt)
        else:
            st.warning("Please enter a prompt to generate an image.")

    # Display the image if it exists in session state
    if st.session_state.image_bytes_gen:
        st.image(st.session_state.image_bytes_gen, caption="Generated Image", use_column_width=True)

        # Only re-extract tags if the image is new
        if st.session_state.tags_gen == []:
            st.session_state.tags_gen = extract_tags_with_llava_image_generation(st.session_state.image_bytes_gen)

        # Generate story based on tags
        if st.session_state.tags_gen:
            genre = st.selectbox("Choose a story genre", ["fantasy", "mystery", "romance", "sci-fi", "adventure"])
            length = st.selectbox("Choose the length of the story", ["short", "medium", "long"])

            if st.button("Generate Story"):
                st.session_state.story_gen = generate_story_with_ollama(st.session_state.tags_gen, genre, length)

            # Display the generated story
            if st.session_state.story_gen:
                st.subheader("Generated Story")
                st.write(st.session_state.story_gen)

                # Option to translate the story
                translate_option = st.checkbox("Translate Story")
                if translate_option:
                    language = st.selectbox("Choose a language", list(LANGUAGE_MODELS.keys()))
                    if st.button("Translate"):
                        model_name = LANGUAGE_MODELS[language]
                        st.session_state.translated_story_gen = translation(language, st.session_state.story_gen, model_name)

                # Display translated story if available
                if st.session_state.translated_story_gen:
                    st.subheader(f"Translated Story ({language})")
                    st.write(st.session_state.translated_story_gen)

                    if st.button("Play Translated Story Audio"):
                        text_to_speech(st.session_state.translated_story_gen, languages_dict[language])
                        st.session_state.show_evaluation_gen = True  # Show evaluation after audio plays

                # Play original story audio
                if st.button("Play Original Story Audio"):
                    text_to_speech(st.session_state.story_gen, "en")
                    st.session_state.show_evaluation_gen = True  # Show evaluation after audio plays

                # Display evaluation section if the evaluation flag is set
                if st.session_state.show_evaluation_gen:
                    st.subheader("General Evaluation")
                    
                    # Display the feedback text area linked to the session state
                    feedback = st.text_area("Provide feedback on the story", value=st.session_state.feedback_text_gen)
                    
                    # When "Send Evaluation" is pressed
                    if st.button("Send Evaluation"):
                        st.write("Thank you for your evaluation!")
                        st.session_state.feedback_text_gen = ""  # Clear feedback text after submission
                        st.session_state.show_evaluation_gen = False  # Optionally hide evaluation after submission


def main():
    page = st.sidebar.selectbox("Choose between Generating or Uploading an Image", ["Upload Image", "Generate Image"])

    if page == "Upload Image":
        uploading_image_main()
    elif page == "Generate Image":
        image_generation_main()

if __name__ == "__main__":
    main()
