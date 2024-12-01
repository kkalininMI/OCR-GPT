# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 00:32:36 2024

@author: Kirill
"""

import difflib
import re
import openai
from PIL import Image, ImageEnhance
from io import BytesIO
import os
import base64
import requests
import pytesseract
from pdf2image import convert_from_path
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import sys
import cv2
import argparse


def encode_image(jpgfile):
    """Encodes an image (JPG) to base64 format."""
    if isinstance(jpgfile, Image.Image):
        buffer = BytesIO()
        jpgfile.save(buffer, format="JPEG")
        jpgfile = buffer.getvalue()
    return base64.b64encode(jpgfile).decode('utf-8')

def reconstruct_text_with_newlines(ocr_data):
    # Group words by line number
    lines = {}
    for word, line_num in zip(ocr_data['text'], ocr_data['line_num']):
        if word.strip():  # Exclude empty words
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(word)

    # Join words in each line with a space and combine lines with a newline
    reconstructed_text = "\n".join(" ".join(words) for words in lines.values())
    return reconstructed_text

def run_tesseract(jpgfile):
    """
    Extracts OCRed text and confidence information from an image.
    """
    # Get detailed OCR data
    ocr_data = pytesseract.image_to_data(jpgfile, config='--psm 6 --oem 3', output_type=pytesseract.Output.DICT)
    
    # Combine all detected words into a single string
    ocr_text = " ".join([word for word in ocr_data['text'] if word.strip()])
    ocr_text = reconstruct_text_with_newlines(ocr_data)
       
    # Extract confidence scores
    confidences = [conf for conf in ocr_data['conf'] if conf != -1]
    
    # Analyze confidence
    confidence_analysis = None
    if confidences:
        confidence_analysis = {
            "average_confidence": sum(confidences) / len(confidences),
            "low_confidence_proportion": sum(1 for conf in confidences if conf < 50) / len(confidences),
            "confidence_std_dev": (sum((x - sum(confidences) / len(confidences)) ** 2 for x in confidences) / len(confidences)) ** 0.5
        }
    
    return ocr_text, confidence_analysis


def analyze_gpt_tokens(response, threshold):
    # Extract token data from the response
    tokens_data = response.get("choices", [])[0]  # Adjusted for actual response
    tokens_info = tokens_data.get('logprobs', [])
    
    content = tokens_info.get('content', [])

    # Extract tokens and logprobs from the content
    tokens = [item['token'] for item in content if 'token' in item]
    logprobs = [item['logprob'] for item in content if 'logprob' in item]
     
    # Initialize the dictionary to store low-confidence token phrases
    low_conf_token_phrases = {}
    
    # Iterate through the tokens and logprobs
    for i, (token, logprob) in enumerate(zip(tokens, logprobs)):
        if logprob < threshold:  # Check if the token is low-confidence
            # Get the slice of 10 tokens before and 10 tokens after
            start_idx = max(0, i - 15)  # Ensure that the start index is not negative
            end_idx = min(len(tokens), i + 16)  # Ensure the end index doesn't go beyond the list
    
            # Construct the phrase by joining the tokens in the slice
            phrase = " ".join(tokens[start_idx:end_idx])
            # Replace double spaces with single spaces
            phrase = " ".join(phrase.split())  # This also removes any leading/trailing spaces
            # Remove extra space before punctuation marks (comma, period, quotes, etc.)
            phrase = re.sub(r'\s([.,;?!])', r'\1', phrase)
            # Record the phrase with the low-confidence token as the key
            low_conf_token_phrases[token] = phrase
    
    # Filter out logprobs that are lower than the threshold
    filtered_logprobs = [logprob for logprob in logprobs if logprob > threshold]
    # filtered_tokens = [tokens[i] for i in range(len(logprobs)) if logprobs[i] > threshold]    
    # Now you can compute perplexity with the filtered logprobs
    perplexity_filtered_outliers = np.exp(-np.mean(filtered_logprobs))
        
    # Calculate confidence for GPT text
    gpt_confidence = calculate_gpt_confidence(filtered_logprobs)
    
    return {
        "low_conf_tokens": low_conf_token_phrases,
        "perplexity": perplexity_filtered_outliers,
        "gpt_confidence": gpt_confidence
    }

def estimated_gpt_costs(response_data, cost_per_1000_tokens):
    """Calculates estimated costs from GPT API response."""
    usage = response_data.get('usage', {})
    total_tokens = usage.get('total_tokens', 0)
    #cost_per_1000_tokens = 0.03  # Example cost for GPT-4 Turbo (check OpenAI pricing)
    cost = (total_tokens / 1000) * cost_per_1000_tokens
    return cost

#jpgfile = page_image, prompt, api_key, token_outlier)

def run_gpt(jpgfile, prompt, api_key, token_outlier, cost_per_1000_tokens):
    """Uses GPT API to process OCR text with image and prompt."""
    base64_image = encode_image(jpgfile)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt_data = {
        "model": "gpt-4-turbo",
        "logprobs": True,
        "top_logprobs": 2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=prompt_data)
    response_message = response.json()
    
    if 'error' in response_message:
        error_message = response_message['error'].get('message', 'An unknown error occurred.')
        print(f"Error: {error_message}")
        sys.exit(1)
    else:
        generated_text = response_message['choices'][0]['message']['content']
        analyzed_tokens = analyze_gpt_tokens(response_message, token_outlier)    
        estimated_cost = estimated_gpt_costs(response_message, cost_per_1000_tokens)

    return generated_text, analyzed_tokens, estimated_cost


def test_gpt(text_fragment, prompt, api_key, language):
    """Uses GPT API to proofread the text fragment based on the given prompt."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    prompt = (
        f"Detect grammatical or lexical inaccuracies, semantic inconsistencies, and incorrect word usage in {language}." 
        "Copy and highlight problematic fragments with asterisks ('<' as the start of the problematic fragment and '>' as the end of the problematic fragment)." 
        "Then, create a short enumerated list (e.g., 1. 2.) of the identified problems.")
    
    # Construct the prompt to send to GPT-3
    prompt_data = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "text",
                        "text": text_fragment
                    }
                ]
            }
        ]
    }

    # Send the request to the GPT API
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=prompt_data)
    
    # Parse the response from GPT
    response_message = response.json()
    generated_text = response_message['choices'][0]['message']['content']
    
    estimated_cost = estimated_gpt_costs(response_message)
    # Return the generated text and the analyzed tokens
    return generated_text, estimated_cost
    
    

def normalized_levenshtein_distance(str1, str2):
    """
    Calculates the normalized Levenshtein distance and similarity score.
    Returns a similarity score between 0 and 1.
    """
    distance = levenshtein_distance(str1, str2)  # Use your existing function
    max_length = max(len(str1), len(str2))
    if max_length == 0:  # Handle edge case for empty strings
        return 1.0  # Identical if both are empty
    normalized_distance = distance / max_length
    similarity_score = 1 - normalized_distance
    return similarity_score


def compare_to_base(base_text, gpt_text):
    """
    Compares OCR text with GPT-generated text using multiple metrics.
    Returns detailed differences, similarity ratio, edit distance, Jaccard similarity, and cosine similarity.
    """
    # Compute differences using difflib
        
    diff = difflib.ndiff(base_text.splitlines(), gpt_text.splitlines())
       
    differences = "\n".join(diff)

    # Calculate similarity ratio
    similarity_ratio = difflib.SequenceMatcher(None, base_text, gpt_text).ratio()

    # Calculate normalized levenshtein distance (Edit Distance)
    normalized_levenshtein = normalized_levenshtein_distance(base_text, gpt_text)

    # Calculate Jaccard Similarity
    ocr_set = set(base_text.split())
    gpt_set = set(gpt_text.split())
    jaccard_similarity = len(ocr_set.intersection(gpt_set)) / len(ocr_set.union(gpt_set)) if ocr_set.union(gpt_set) else 0.0

    # Calculate Cosine Similarity
    vectorizer = CountVectorizer().fit_transform([base_text, gpt_text])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0, 1]

    return {
        "differences": differences,
        "similarity_ratio": similarity_ratio,
        "normalized_levenshtein": normalized_levenshtein,
        "jaccard_similarity": jaccard_similarity,
        "cosine_similarity": cosine_sim
    }



def calculate_gpt_confidence(log_probs):
    """
    Calculate confidence for GPT OCR text based on log probabilities.
    Args:
        log_probs (list): Log probabilities of tokens from GPT output.

    Returns:
        float: Average confidence score.
    """
    import numpy as np
    probabilities = [np.exp(log_prob) for log_prob in log_probs]  # Convert log probabilities to probabilities
    return np.mean(probabilities)


def unified_confidence(log_probs, perplexity):
    """
    Combine token probabilities and perplexity to compute a unified confidence score.
    Args:
        log_probs (list): Log probabilities of tokens.
        perplexity (float): Perplexity score.

    Returns:
        float: Unified confidence score.
    """
    token_confidence = calculate_gpt_confidence(log_probs)
    perplexity_confidence = 1 / (1 + perplexity)  # Normalize perplexity to a confidence-like scale
    return (token_confidence + perplexity_confidence) / 2  # Average the two

#filename = "80040_0000a_trans-3-4.pdf"

# Define enhancement functions
def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image, 5)

def enhance_image_grayscale_sharpen(image):
    open_cv_image = np.array(image)
    open_cv_image = get_grayscale(open_cv_image)
    open_cv_image = remove_noise(open_cv_image)
    open_cv_image = sharpen_image(open_cv_image)
    im_pil = Image.fromarray(open_cv_image)
    return im_pil

def enhance_image_thresholding(image):
    open_cv_image = np.array(image)
    open_cv_image = get_grayscale(open_cv_image)
    open_cv_image = remove_noise(open_cv_image)
    open_cv_image = cv2.threshold(open_cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    im_pil = Image.fromarray(open_cv_image)
    return im_pil

def enhance_image_sharpness(image):
    im_out = ImageEnhance.Sharpness(image).enhance(5)
    return im_out

def enhance_image_contrast(image):
    im_out = ImageEnhance.Contrast(image).enhance(10)
    return im_out

def enhance_image_brightness(image):
    im_out = ImageEnhance.Brightness(image).enhance(10)
    return im_out

def adaptive_threshold(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresholded_image)

def deskew_image(image):
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = open_cv_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(open_cv_image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

# Define function to run Tesseract with confidence
def run_tesseract_with_confidence(image):
    ocr_data = pytesseract.image_to_data(image, config='--psm 6 --oem 3', output_type=pytesseract.Output.DICT)
    return ocr_data

# Function to analyze the confidence scores
def analyze_confidence(ocr_data):
    confidences = [
        int(conf) for conf in map(str, ocr_data['conf']) if conf.isdigit()
    ]
    return {
        'mean_confidence': sum(confidences) / len(confidences) if confidences else None,
        'min_confidence': min(confidences) if confidences else None,
        'max_confidence': max(confidences) if confidences else None,
    }

# Function to select the best enhancement
def select_best_enhancement(image):
    # List of enhancement functions to test
    enhancement_functions = [
        ('default_enhancement', lambda x: x),  # Default enhancement (no changes)
        ('enhance_image_grayscale_sharpen', enhance_image_grayscale_sharpen),
        ('enhance_image_thresholding', enhance_image_thresholding),
        ('enhance_image_sharpness', enhance_image_sharpness),
        ('enhance_image_contrast', enhance_image_contrast),
        ('enhance_image_brightness', enhance_image_brightness),
        ('adaptive_threshold', adaptive_threshold),
        ('deskew_image', deskew_image),
    ]
    
    best_confidence = -1  # Initialize with a very low confidence score
    best_enhancement_name = None
    best_ocr_data = None
    
    # Test each enhancement function and select the one with the best confidence score
    for name, enhance_func in enhancement_functions:
        enhanced_image = enhance_func(image)
        ocr_data = run_tesseract_with_confidence(enhanced_image)
        confidence_analysis = analyze_confidence(ocr_data)
        
        print(f"Confidence analysis for {name}: {confidence_analysis}")
        
        # Compare the mean confidence score
        if confidence_analysis['mean_confidence'] is not None and confidence_analysis['mean_confidence'] > best_confidence:
            best_confidence = confidence_analysis['mean_confidence']
            best_enhancement_name = name
            best_ocr_data = ocr_data
    
    print(f"Best enhancement function: {best_enhancement_name} with mean confidence: {best_confidence}")
    return best_enhancement_name, best_ocr_data



def implement_ocr(filename, compare_gpt=False, 
                  page_num=True, api_key=None,
                  language="English",
                  token_outlier=-2, cost_per_1000_tokens=0.03, gpt_test=False, 
                  enhance=False):
    """
    Processes a PDF to perform OCR using GPT and compare with either Tesseract OCR or GPT-based OCR.
    Outputs a dictionary with metrics and GPT OCR texts, with optional image enhancement.

    Args:
        filename (str): Path to the PDF file.
        comparison_mode (str): "tesseract" or "gpt_ocr".
        api_key (str): OpenAI API key for GPT-based OCR.
        enhance (bool): If True, runs image enhancement before OCR.
    
    Returns:
        dict: Contains metrics with pages as columns and GPT OCR texts.
    """
    print(f"Processing {filename}")
    txtfilename = filename.replace('.pdf', '_ocrgpt.txt')

    # Store results
    gpt_texts = []
    gpt_cost = []
    gpt_tokens = {}
    gpt_tests = {}
    
    est_cost_b = 0
    est_cost_s = 0
    
    metrics = {
        "Similarity Ratio GPT OCR vs. Tesseract": [],
        "Normalized Levenshtein GPT OCR vs. Tesseract": [],
        "Jaccard Similarity GPT OCR vs. Tesseract": [],
        "Cosine Similarity GPT OCR vs. Tesseract": [],
        "Similarity Ratio GPT OCR vs. GPT Base": [],
        "Normalized Levenshtein GPT OCR vs. GPT Base": [],
        "Jaccard Similarity GPT OCR vs. GPT Base": [],
        "Cosine Similarity GPT OCR vs. GPT Base": [],
        "Perplexity GPT OCR": [],
        "Perplexity GPT Base": [],
        "Average Prob. GPT OCR": [],
        "Average Conf. Tesseract": [],
        "Average Prob. GPT Base": [],
        "Estimated GPT OCR Cost": []
    }

    # Convert PDF pages to images
    pages = convert_from_path(filename, dpi=300)

    # Determine the best enhancement function if enhancement is enabled
    best_enhancement_func = None
    if enhance:
        print("Running enhancement selection on the first page...")
        best_enhancement_func_name, _ = select_best_enhancement(pages[0])
        print(f"Best enhancement function selected: {best_enhancement_func_name}")
        
        # Retrieve the actual function based on the name
        enhancement_map = {
            "default_enhancement": lambda x: x,
            "enhance_image_grayscale_sharpen": enhance_image_grayscale_sharpen,
            "enhance_image_thresholding": enhance_image_thresholding,
            "enhance_image_sharpness": enhance_image_sharpness,
            "enhance_image_contrast": enhance_image_contrast,
            "enhance_image_brightness": enhance_image_brightness,
            "adaptive_threshold": adaptive_threshold,
            "deskew_image": deskew_image,
        }
        best_enhancement_func = enhancement_map[best_enhancement_func_name]

    with open(txtfilename, 'w', encoding='utf-8') as f:
        for page_number, page_image in enumerate(pages, start=1):
            
            average_confidence_tess = np.nan
            average_gpt_confidence_base = np.nan
            average_gpt_confidence_ocr = np.nan
            base_tokens_perplexity = np.nan


            # Apply the best enhancement function if selected
            if enhance and best_enhancement_func:
                page_image = best_enhancement_func(page_image)

            # Generate text with GPT
            prompt = (
                f"Please transcribe this page in {language} accurately. "
                "Replace any hard-to-recognize words with your best guess. "
                "Mark each new line with \n."
                "Any comments from the model about the implementation of this task are prohibited. "
                "Your message should only contain text from the image."
            )
            gpt_text, tokens, est_cost = run_gpt(page_image, prompt, api_key, token_outlier, cost_per_1000_tokens)
            average_gpt_confidence_ocr = round(tokens['gpt_confidence'], 3) * 100
            gpt_texts.append(gpt_text)
            gpt_cost.append(est_cost)

            gpt_tokens["Page " + str(page_number)] = tokens['low_conf_tokens']

            if page_num:
                f.write("\n")
                f.write(f"### Page {page_number} ###\n")  # Inserts the page number
                f.write("\n")        
            
            f.write(f"{gpt_text}\n")

            # Generate base text
            if compare_gpt:
                base_prompt = ("Please transcribe this page accurately. " 
                               "Replace any hard-to-recognize words with '?????'.")    
                base_text_gpt, base_tokens, est_cost_b = run_gpt(page_image, base_prompt, api_key, token_outlier, cost_per_1000_tokens)
                average_gpt_confidence_base = round(base_tokens['gpt_confidence'], 3) * 100
                base_tokens_perplexity = base_tokens['perplexity']
                 
            base_text_tess, confidence_tess = run_tesseract(page_image)
            average_confidence_tess = confidence_tess['average_confidence']
        
            compare_to_tess = compare_to_base(base_text_tess, gpt_text)            
            
            if compare_gpt:
                compare_to_gpt = compare_to_base(base_text_gpt, gpt_text)
            else:
                compare_to_gpt = {
                    "differences": np.nan,
                    "similarity_ratio": np.nan,
                    "normalized_levenshtein": np.nan,
                    "jaccard_similarity": np.nan,
                    "cosine_similarity": np.nan
                    }
            
            metrics["Similarity Ratio GPT OCR vs. Tesseract"].append(compare_to_tess["similarity_ratio"])
            metrics["Normalized Levenshtein GPT OCR vs. Tesseract"].append(compare_to_tess["normalized_levenshtein"])
            metrics["Jaccard Similarity GPT OCR vs. Tesseract"].append(compare_to_tess["jaccard_similarity"])
            metrics["Cosine Similarity GPT OCR vs. Tesseract"].append(compare_to_tess["cosine_similarity"])
            metrics["Similarity Ratio GPT OCR vs. GPT Base"].append(compare_to_gpt["similarity_ratio"])
            metrics["Normalized Levenshtein GPT OCR vs. GPT Base"].append(compare_to_gpt["normalized_levenshtein"])
            metrics["Jaccard Similarity GPT OCR vs. GPT Base"].append(compare_to_gpt["jaccard_similarity"])
            metrics["Cosine Similarity GPT OCR vs. GPT Base"].append(compare_to_gpt["cosine_similarity"])
            metrics["Perplexity GPT OCR"].append(tokens['perplexity'])
            metrics["Perplexity GPT Base"].append(base_tokens_perplexity)
            metrics["Average Prob. GPT OCR"].append(average_gpt_confidence_ocr)
            metrics["Average Conf. Tesseract"].append(round(average_confidence_tess, 3))
            metrics["Average Prob. GPT Base"].append(average_gpt_confidence_base)
            metrics["Estimated GPT OCR Cost"].append(est_cost)
            
    # Convert metrics to DataFrame-like structure
    metrics_df = pd.DataFrame(metrics, index=[f"Page {i+1}" for i in range(len(pages))]).T
    metrics_df["Average"] = metrics_df.mean(axis=1)
    metrics_df["Std.Dev"] = metrics_df.std(axis=1)

    gpt_ocr = " ".join(gpt_texts)
    
    if gpt_test:
        for i, text_fragment in enumerate(gpt_texts):
            response, est_cost_t = test_gpt(text_fragment, prompt, api_key, language)
            page_number = i + 1  
            gpt_tests["Page " + str(page_number)] = response
            est_cost_s = est_cost_s + est_cost_t
            
    est_cost_sum = sum(gpt_cost)
    total_cost = est_cost_sum + est_cost_s + est_cost_b
    
    total_cost = f"Total cost: ${round(total_cost, 2)} = GPT_OCR(${round(est_cost_sum, 2)}) + GPT_Base(${round(est_cost_b, 2)}) + GPT_Test(${round(est_cost_s, 2)})"
        
    # Final output dictionary
    result = {
        "gpt_ocr_text": gpt_ocr,
        "metrics": metrics_df,
        "lowest_logprob_tokens": gpt_tokens,
        "gpt_test": gpt_tests,
        "total_cost": total_cost
    }
    return result

def implement_ocr_list(file_list, **kwargs):
    
    results = []
    
    for file in file_list:
        
        filename = os.path.basename(file)
        
        ocr_output = implement_ocr(file, **kwargs)
        
        metrics_df = ocr_output['metrics']
        averages = metrics_df["Average"].tolist()[:-1]
        averages = [round(i,3) for i in averages]
        
        cost_ext = re.search(r'Total cost: \$([0-9]*\.?[0-9]+)', ocr_output["total_cost"])
        total_cost = float(cost_ext.group(1)) if cost_ext else None
        
        result = {
            "filename": filename,
            "total_cost": total_cost
        }
        
        for idx, avg in enumerate(averages, start=1):
            result[f"average_{idx}"] = avg
        
        results.append(result)
    
    summary_df = pd.DataFrame(results)
    
    colnames = [
       "filename", "Est.Total Cost", "Sim GPT-Tess",
       "Lev GPT-Tess", "Jac GPT-Tess", "CosSim GPT-Tess",
       "SimR GPT-Base", "Lev GPT-Base", "Jac GPT-Base",
       "CosSim GPT-Base", "Perp GPT", "Perp Base",
       "Av.Prob.GPT", "Av.Conf.Tess", "Av.Prob.Base"]
   
    if summary_df.shape[1] == len(colnames):
        summary_df.columns = colnames
    else:
        print("Warning: Number of columns does not match the specified column names.")
        
    # Add the total row
    total_cost_sum = summary_df["Est.Total Cost"].sum()
    total_row = pd.DataFrame([["Total.Cost.All", total_cost_sum] + [None] * (len(summary_df.columns) - 2)],
                             columns=summary_df.columns)
    
    # Append the total row to the summary DataFrame
    summary_df = pd.concat([summary_df, total_row], ignore_index=True)
       
    return summary_df


def implement_ocr_folder(folder_path, **kwargs):
    """
    Processes all PDF files in the specified folder path using OCR and generates a summary.

    Args:
        folder_path (str): Path to the folder containing PDF files.
        **kwargs: Additional arguments to pass to `implement_ocr_list`.

    Returns:
        pd.DataFrame: Summary DataFrame containing results for all processed files.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided path '{folder_path}' is not a valid directory.")

    # Get all PDF files in the folder
    file_list = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith('.pdf')
    ]

    if not file_list:
        raise ValueError(f"No PDF files found in the directory '{folder_path}'.")

    # Call the implement_ocr_list function
    summary_df = implement_ocr_list(file_list, **kwargs)

    return summary_df


class ParseTrueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() in ('true', 't', 'yes', 'y', '1'):
            setattr(namespace, self.dest, True)
        elif values.lower() in ('false', 'f', 'no', 'n', '0'):
            setattr(namespace, self.dest, False)
        else:
            raise argparse.ArgumentError(self, "Invalid boolean value")



def main():
    parser = argparse.ArgumentParser(description='OCR Tool with GPT and Tesseract Comparison')

    # Common arguments
    parser.add_argument('--filename', help='Path to the PDF file (for single file processing)')
    parser.add_argument('--file_list', nargs='+', help='List of PDF files (for batch processing)')
    parser.add_argument('--folder_path', help='Path to a folder containing PDF files (for folder processing)')
    parser.add_argument('--compare_gpt', action=ParseTrueAction, default=False, 
                        help='Whether to compare GPT-based OCR with another method (true/false)')
    parser.add_argument('--page_num', action=ParseTrueAction, default=True, 
                        help='Whether to include page numbers in output (true/false)')
    parser.add_argument('--api_key', default=None, help='OpenAI API key for GPT-based OCR')
    parser.add_argument('--language', type=str, default='English',
                       help='Specify the language for OCR processing (default: English)')
    parser.add_argument('--token_outlier', type=int, default=-2, 
                        help='Threshold for outlier token handling in GPT OCR')
    parser.add_argument('-cost_per_1000_tokens', type=int, default=0.03, 
                        help='Cost per 1000 tokens')   
    parser.add_argument('--gpt_test', action=ParseTrueAction, default=False, 
                        help='Whether to run additional GPT tests (true/false)')
    parser.add_argument('--enhance', action=ParseTrueAction, default=False, 
                        help='Whether to apply image enhancement before OCR (true/false)')
    parser.add_argument('--mode', choices=['single', 'batch', 'folder'], required=True,
                        help='Choose "single" for single file processing, "batch" for batch processing, or "folder" for folder processing')

    args = parser.parse_args()

    if args.mode == 'single':
        if not args.filename:
            raise ValueError("Filename is required for single file processing mode.")
        
        # Call the implement_ocr function for single file processing
        result = implement_ocr(
            filename=args.filename,
            compare_gpt=args.compare_gpt,
            page_num=args.page_num,
            api_key=args.api_key,
            language=args.language,
            token_outlier=args.token_outlier,
            cost_per_1000_tokens=args.cost_per_1000_tokens,
            gpt_test=args.gpt_test,
            enhance=args.enhance
        )
        print(result)
    
    elif args.mode == 'batch':
        if not args.file_list:
            raise ValueError("File list is required for batch processing mode.")
        
        # Call the implement_ocr_list function for batch processing
        summary = implement_ocr_list(
            file_list=args.file_list,
            compare_gpt=args.compare_gpt,
            page_num=args.page_num,
            api_key=args.api_key,
            language=args.language,
            token_outlier=args.token_outlier,
            cost_per_1000_tokens=args.cost_per_1000_tokens,
            gpt_test=args.gpt_test,
            enhance=args.enhance
        )
        print(summary)
    
    elif args.mode == 'folder':
        if not args.folder_path:
            raise ValueError("Folder path is required for folder processing mode.")
        
        # Call the implement_ocr_folder function for folder processing
        summary_df = implement_ocr_folder(
            folder_path=args.folder_path,
            compare_gpt=args.compare_gpt,
            page_num=args.page_num,
            api_key=args.api_key,
            language=args.language,
            token_outlier=args.token_outlier,
            cost_per_1000_tokens=args.cost_per_1000_tokens,
            gpt_test=args.gpt_test,
            enhance=args.enhance
        )
        print(summary_df)

if __name__ == "__main__":
    main()