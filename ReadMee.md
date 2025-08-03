Automated Story Generation from Images
Introduction
This project addresses the challenge of generating coherent and contextually relevant stories from a sequence of images. We have developed a system that uses a combination of deep learning models to first generate captions for individual images and then weave these captions into a narrative story. The project's primary goal is to enhance narrative coherence at both local and global levels, ensuring the generated text is both readable and consistent.

Project Flow
The system operates in a sequential, two-step process:

Image Caption Generation: We use a Convolutional Neural Network (CNN) combined with an Long Short-Term Memory (LSTM) model to analyze images and produce descriptive captions. The CNN extracts visual features from the images, which are then processed by the LSTM to form natural language sentences.

Story Generation: The generated captions from the first step are fed into a fine-tuned GPT-2 model. This powerful language model takes the captions as a starting point and expands upon them to create a coherent, multi-sentence story.

Models and Implementation
1. Image Captioning Model (CNN + LSTM)
The image captioning model is built using the VGG16 architecture for feature extraction and a custom encoder-decoder model with LSTM layers for sequence generation.

Encoder Model: The encoder is a pre-trained VGG16 model from tensorflow.keras.applications. We remove the final classification layer and use the output of the fc2 layer as the image feature vector.

Decoder Model: The decoder is a custom network that takes the image features and a sequence of text as input. It consists of:

A Dropout layer on the image features.

A Dense layer with relu activation.

An Embedding layer for the text input.

An LSTM layer to process the text sequence.

A Dense layer with softmax activation for predicting the next word.

The model is compiled with loss='categorical_crossentropy' and optimizer='adam'.

2. Story Generation Model (GPT-2)
A GPT-2 language model from the transformers library is used for story generation. This model is fine-tuned on a corpus of stories to improve its narrative capabilities.

Tokenizer: GPT2Tokenizer.from_pretrained("gpt2") is used.

Model: GPT2LMHeadModel.from_pretrained("gpt2") is used.

Fine-tuning: The model is fine-tuned on a custom dataset, presumably the allcaps.txt file mentioned in the notebook. This fine-tuning process helps the model adapt to the style and structure of storytelling.

Generation Parameters: Stories are generated using parameters like max_length=100, num_beams=5, and no_repeat_ngram_size=2 to ensure quality and prevent repetitive output.

Datasets
The project utilizes the following datasets for training and evaluation:

Flickr 8k Dataset: This dataset, consisting of 8,000 images, each with five captions, is used to train the CNN + LSTM model for image caption generation. The images were preprocessed to extract features using the VGG16 model.

ROCStories Corpus: This corpus of five-sentence stories is used to train the GPT-2 model. It is valuable for its focus on common-sense narratives and causal relationships, which helps in generating coherent stories.

VIST Dataset (Visual Storytelling Dataset): This dataset was explored but found to be largely inaccessible due to its large size and access denial issues. However, a pre-trained model (ReCo-RL) was used to evaluate the outputs, which provided valuable insights into the coherence of the generated stories.

Evaluation
The coherence and quality of the generated stories were evaluated using the BLEU score, a metric commonly used in machine translation and text generation tasks.

BLEU-1 Score: 0.101141

BLEU-2 Score: 0.318026

The report notes that a BLEU-1 score of 0.1 is low, indicating a low unigram overlap with the reference text. A higher BLEU-2 score of 0.31 suggests a better bigram overlap, showing that the model is better at generating two-word sequences that match the reference stories.

Dependencies
The project relies on the following key libraries, primarily from the Python ecosystem:

tensorflow

keras

numpy

tqdm

transformers

torch

nltk

PIL

matplotlib

re

Issues Encountered
The development process was not without its challenges. Several dependencies and datasets proved difficult to work with:

Brown Coherence Package: An essential package for extracting entity features, it was found to be unavailable online with no suitable replacement that was compatible with the CRCN model.

PyTorch Dependencies: The project required an older version of PyTorch (1.6.0), which led to "No matching distribution found" errors.

VIST and SSID Datasets: These datasets were either too large, required special access that was denied, or were completely unavailable. This limited the ability to fine-tune some of the models as originally planned.