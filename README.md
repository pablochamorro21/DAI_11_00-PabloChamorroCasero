# Image Tagging and Translation Pipeline

## Project Overview

This project, presented as a Streamlit, automates the generation of images and stories based on user input, with customizable options for image generation, tagging, and storytelling. Users can either generate an image based on a descriptive prompt or upload an existing image. After the image is created or uploaded, the system automatically generates five descriptive tags using OLLAMA. The user can then select their preferred story genre and length. Based on these choices, the project generates a custom story.

Once the story is created, users have the option to either listen to it in English or translate and listen to it in another language. The project employs pre-trained models from several machine learning libraries, all within a Streamlit-based interface designed for ease of use. The code structure and documentation are organized to provide clear, step-by-step guidance, making the project’s design, logic, and decisions easy to follow.

### Key Functionalities

1. **Image Upload, Prompt-based Image Generation, and Tag Generation**:
    - Users can either upload an image or generate one based on a custom prompt. The images are processed using the LLaVA model via `ollama`, which generates five descriptive tags to capture key aspects of the image's content and style.
2. **Story Generation and Customization**:
    - After tags are created, users can select parameters for a story based on the image, such as genre (e.g., fantasy, sci-fi, mystery…) and length (short, medium, long). The project then generates a unique story that aligns with these specifications, bringing the image to life with a creative narrative.
3. **Multilingual Translation Pipeline**:
    - Pre-trained MarianMT models support translation of generated tags, stories, or text into a range of languages including Spanish, French, German, and Portuguese, making the content accessible for multilingual users.
4. **Text-to-Speech (TTS) Functionality**:
    - Users can listen to the generated story, either in its original English or in any translated language selected. A TTS model, gTTs, converts the text to audio, allowing users to enjoy the narrative in their language of choice. Furthermore, the library pydub, with the model AudioSegment enables the user to choose the speed of the person telling the story.
5. **Interactive User Interface**:
    - Streamlit provides an easy-to-use interface where users can upload images, view descriptions and tags, and request translations.

---

## Project Structure and Files

- **`individual_project.py`**: A file that serves as the central code module for the project, housing key functionalities such as image processing, image generation, tag generation, story creation, multilingual translation, and text-to-speech. This script coordinates all backend operations, integrating the various ML models and managing the interactions between them to provide a seamless experience.
- **`requirements.txt`**: A list of all dependencies to run the code.
- **`README.md`:**  A file that provides a comprehensive overview of the project, detailing its purpose, functionalities, and usage instructions. It outlines the goals of the project, such as generating images and stories based on user input, creating descriptive tags, and providing multilingual translation and text-to-speech options. The README includes setup instructions, guides for installing dependencies, and an explanation of each feature to help users understand and navigate the project. Additionally, it includes examples, notes on model usage, and instructions for interacting with the Streamlit-based interface, making it a central reference for new users and contributors alike.

---

## Installation and Setup

To run this project, please follow these steps:

1. **Clone the repository** (if applicable).
2. **Install dependencies**:
    
    ```

    pip install -r requirements.txt
    
    ```
    
3. **Run the Streamlit Application**:
    
    ```
    
    streamlit run individual_project.py
    
    ```
    
4. **Requirements File**: Lists libraries such as `streamlit`, `transformers`, `ollama`, and others that need to be installed. Specific instructions for newer packages or uncommon dependencies are included.

---

## Design Approach

The project follows the steps outlined in the assignment approach:

### 1. Problem Understanding

The project focuses on providing a streamlined pipeline for processing images to extract tags, translating them, generating an audio with the story, and facilitating this through a web-based UI. The main goal is to automate image description and multilingual support to help a broader user base understand image content.

### 2. Problem Decomposition

- **Image Upload & Processing**: Implemented using `streamlit` and `PIL` for image handling.
- **Tag Extraction**: Utilizes the LLaVA model for generating relevant tags, integrated with `ollama` API calls.
- **Translation Pipeline**: Uses pre-trained MarianMT translation models from Hugging Face for multilingual support.
- **Image Upload & Processing**: Handled via `streamlit` for the user interface and `PIL` (Python Imaging Library) for image manipulation and processing, allowing users to either upload an image or generate one from a prompt.
- **Tag Extraction**: Integrated with the LLaVA model through `ollama` API calls to generate descriptive tags. These tags capture essential details about the image, aiding in story generation and enhancing user customization options.
- **Story Generation**: Uses `ollama` to create a unique story based on the image tags, with customizable options for genre and length. This story generation module draws on the tags and selected parameters to create a narrative that aligns with the image’s themes.
- **Translation Pipeline**: Leverages pre-trained MarianMT models from Hugging Face to provide multilingual translation of tags and generated stories. This feature supports several languages, including Spanish, French, German, and Portuguese, making the content accessible to a broader audience.
- **Text-to-Speech (TTS)**: Converts generated stories into audio files using the model gTTS, allowing users to listen to the story in the original language or any translated language of choice, and enabling the user to choose the speed of the storytelling itself.
- **Interactive UI Elements**: Streamlit's interactive features are used to control the workflow, allowing users to upload images, select story options, view tags, and listen to or download the story and audio. This makes the interface intuitive and accessible for users of all backgrounds.

### 3. Research and Model Selection

- **LLaVA (Large Language and Vision Assistant)**: Selected for its robust tagging and descriptive capabilities, LLaVA generates contextually relevant tags for each image, facilitating enhanced story generation based on visual content.
- **MarianMT Models**: Chosen due to their efficiency and extensive multilingual support, MarianMT provides high-quality translations of tags and stories in multiple languages, including Spanish, French, German, and Portuguese, expanding the project’s accessibility.
- **GTTs (Google Text-to-Speech)**: An additional TTS option, GTTs provides quick and reliable text-to-speech conversion, particularly useful for generating audio in multiple languages. It supports the playback of the story in either the original language or the translated version, offering a high-quality listening experience.
- **Streamlit for UI**: Utilized to create an interactive, user-friendly interface, Streamlit integrates seamlessly with the backend processes, guiding users through each step from image upload to audio playback.

### 4. Pipeline Implementation

**1. Image Upload and Display**

- **Description**: Users can upload an image to be processed by the application.
- **Steps**:
    - **Upload**: The application accepts image files uploaded by the user.
    - **Display**: The uploaded image is displayed within the UI using Streamlit for easy viewing and verification before processing.

**2. Image Generation (Optional)**

- **Functionality**: The application includes an option to generate images using the Stable Diffusion model if users wish to create new images rather than upload existing ones.
- **Steps**:
    - **Prompt Input**: Users can provide a textual prompt to guide the image generation.
    - **Image Generation**: Using the Stable Diffusion model (via `diffusers`), an image is generated based on the provided prompt.
    - **Display**: The generated image is displayed within the UI, and users can choose to proceed with tag generation on this new image.

**3. Tag Generation**

- **Function**: `extract_tags_with_llava(image)`
- **Steps**:
    - Upload and display the image.
    - Save the image temporarily for processing.
    - Call the LLaVA model via `ollama` to generate a description and suggest five relevant tags.
    - Display generated tags in the Streamlit UI.

**4. Translation**

- **Functionality**:
    - A dictionary of MarianMT models is used for translating tags from English into Spanish, French, German, and Portuguese.
    - Each model is mapped to its respective language and initialized for use.

**5. UI Interaction**

- **Interface**:
    - Users can upload images, view generated tags, and request translations.
    - The application provides clear feedback on translation success or failure, and users can refine tags interactively if needed.

**6. Audio** 

- **Feature**:
    - Optional audio to listen to the story using Google Text-to-Speech (gTTS) and PyGame for playback.
    - Users can click to hear generated story in their preferred language.
    - Users can choose the speed of the storytelling thanks to AudioSegment.

**7. Feedback Collection**

- **Functionality**: The application includes a feedback system to capture user input on the quality of tags and translations.
- **Steps**:
    - **User Prompt**: After tags are generated and translations are provided, users are asked to rate the relevance and accuracy of the tags and translations.
    - **Feedback Collection**: Users can provide comments on various aspects, including tag accuracy, translation quality, and UI usability.

This feedback mechanism is essential for iterative improvement and allows for adjustments based on real user insights.

**Pipeline Flowchart**
![Flowchart](flowchart.png)


### 5. Metrics and Evaluation

Evaluation focuses on:

- **Tag Accuracy**: Ensuring tags generated are relevant to the image content.
- **Story generation**: ensuring that the story makes sense, and is in accordance with the tags retrieved
- **Translation Quality**: Verifying the coherence and accuracy of translations.
- **User Feedback**: Collecting feedback from users to refine the tagging and translation experience.

### 6. User Testing

The application was tested with multiple users to evaluate:

- **UI Usability**: Evaluating ease of navigation, layout clarity, and overall user experience.
- **Tag Relevance and Clarity**: Assessing the quality and accuracy of tags generated by the LLaVA model.
- **Translation Accuracy and Appropriateness**: Reviewing the accuracy, contextual appropriateness, and clarity of translations across supported languages.
- **Feature Effectiveness**: Testing advanced functionalities, such as text-to-speech and image generation from text prompts, to gauge their usability and impact.

To further understand user experience, feedback was collected through interviews with test users. This qualitative feedback helped highlight areas of improvement and validate the effectiveness of the implemented features.

Feedback:
 
    "The layout is intuitive, and the tagging process is straightforward. I like how easy it is to upload an image and see the results immediately. However, the generation of the story seems to be a bit slow.”


    "The layout is intuitive, and the tagging process is straightforward. I like how easy it is to upload an image and see the results immediately. However, the generation of the story seems to be a bit slow.”
    
    "The Spanish translations were clear and contextually accurate. The system handled idiomatic expressions well.”
    
    "The audio feedback in Spanish was a nice touch, especially as it enables you to do several other things while listening to the story itself. It made the tool feel more interactive.”
    
    "I liked being able to generate images from text—it added an extra layer of creativity. The quality of generated images was impressive. However, the image generation was a bit slow”
    
    "The prompts for the image generation seem to be very effective, and the image generated is quite accurate based on the prompt, obviously taking into account it’s still a machine learning algorithm .”

The feedback gathered through these interviews provided valuable insights, leading to actionable improvements in the application's UI and feature set.

---

## Documentation and Code Structure

The code follows a verbose style, with each function and step documented for ease of understanding. Decision points are commented to clarify why certain models or configurations were chosen.

### Major Functions

1. **`extract_tags_with_llava(image)`**:
    - **Purpose**: Uses the LLaVA model to describe an uploaded image and generate five single-word tags.
    - **Steps**:
        - Temporarily saves the uploaded image file to a designated path.
        - Displays the image in the UI using Streamlit.
        - Calls the LLaVA model via `ollama.chat` to generate a description and extract tags.
        - Parses the response to retrieve the tags, which are then displayed to the user.
        - Deletes the temporary image file after processing to clean up storage.
    - **Error Handling**: Catches and displays any errors encountered during image processing or API response.
2. **`generate_story_with_ollama(tags, genre, length)`**:
    - **Purpose**: Generates a story based on provided tags, genre, and desired length.
    - **Steps**:
        - Builds a prompt incorporating the specified genre, tags, and word count (determined by length).
        - Sends the prompt to the model using `ollama.chat`, streaming the response chunks.
        - Aggregates chunks to form the full story.
    - **Error Handling**: Displays a user-friendly error message if the model request fails.
3. **`word_count(story)`**:
    - **Purpose**: Counts the words in a given story to ensure it aligns with specified length requirements.
    - **Steps**:
        - Splits the story text into words and returns the word count.
4. **`translation(language, story, model_name)`**:
    - **Purpose**: Translates a story into a specified language using MarianMT models.
    - **Steps**:
        - Initializes the tokenizer and model for the specified language.
        - Splits the story into paragraphs and further into manageable chunks to avoid exceeding the model’s input limit.
        - Translates each chunk and reassembles them into full paragraphs, then into the complete translated story.
    - **Error Handling**: Handles large input sizes by breaking paragraphs into sentences and limits token length to avoid errors.
5. **`text_to_speech(text, language)`**:
    - **Purpose**: Converts text to speech in a specified language using Google Text-to-Speech (gTTS) and plays the audio using PyGame.
    - **Steps**:
        - Generates an audio file for the text using gTTS and saves it temporarily.
        - Received the speed chosen by the user to play the audio which includes the story.
        - Initializes PyGame to play the audio file, providing auditory feedback to the user.
        - Cleans up by deleting the temporary audio file after playback.
    - **Error Handling**: Provides error feedback if there are issues with audio generation or playback.
6. **`generate_image_from_text(prompt)`**:
    - **Purpose**: Generates an image based on a text prompt using the Stable Diffusion model.
    - **Steps**:
        - Loads the Stable Diffusion model and directs it to generate an image from the provided prompt.
        - Saves the generated image to a byte array for display in the UI or further processing.
    - **Device-Specific Note**: Configured to use Apple’s MPS device for optimal performance on Apple Silicon.
7. **`extract_tags_with_llava_image_generation(image_bytes)`**:
    - **Purpose**: Similar to `extract_tags_with_llava(image)`, but for handling image data directly from byte streams instead of file paths.
    - **Steps**:
        - Temporarily saves the image bytes to a file, located on the desktop for user access.
        - Displays the image in the UI and starts a timer for performance tracking.
        - Calls LLaVA to describe the image and extract tags, handling any responses or errors.
        - Cleans up the temporary file after processing.
    - **Error Handling**: Includes checks to confirm a valid model response, with error messages for unsuccessful API interactions.

### Enhancements and Value-Added Functionality

To exceed basic functionality requirements, additional features were considered, including:

- **User-Friendly Translation Display**: Translations are shown alongside the original text, with feedback indicators to highlight successful or unsuccessful translation attempts.
- **Interactive Tag Refinement**: Allows users to modify suggested tags to improve relevance and accuracy.
- **Error Handling and Resilience**: Comprehensive error handling provides clear feedback if any part of the pipeline (such as image processing, story generation, or translation) encounters issues. This enhances the reliability of the application and maintains a positive user experience.

---

## Conclusion

This README provides a comprehensive summary of the Image Tagging, Translation, and Generation Pipeline project, covering its purpose, structure, and features. The project meets all requirements and includes key features such as image tagging, translation, story generation, and user feedback collection. Advanced functionalities, like image generation from text prompts using Stable Diffusion and text-to-speech playback using Google Text-to-Speech (gTTS), enhance accessibility and interactivity. The project also includes extensive error handling, a user-friendly interface, and supports various media types, making it a flexible tool for content creation and translation.

GITHUB LINK: https://github.com/pablochamorro21/DAI_11_00-PabloChamorroCasero
