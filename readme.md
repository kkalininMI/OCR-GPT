---
title: "OCR-GPT Package Documentation"
author: "Kirill Kalinin"
date: "2024-11-30"
output: html_document
---

## Overview

This package provides tools to perform Optical Character Recognition (OCR) using GPT models, alongside comparisons with Tesseract OCR and GPT-based OCR alternatives. It focuses on minimizing the risks of hallucinations associated with GPT models by implementing several key features: image enhancement, text comparison metrics, anomaly detection, and cost estimation. The package is designed for both technical users and researchers who require robust OCR performance with cost-efficient execution.

---

## Key Features

### 1. **Image Enhancement**

The package offers several image enhancement techniques to improve OCR accuracy. Enhancements are applied to the first page to determine the best method for the document, which is then used for all pages if enabled (`enhance=True`). The enhancement functions include:

- **Default Enhancement**: No modifications are applied to the image.
- **Grayscale and Sharpen**: Converts the image to grayscale and applies sharpening filters.
- **Thresholding**: Applies thresholding to emphasize text contrasts.
- **Sharpness Enhancement**: Increases the sharpness of the image for better text clarity.
- **Contrast Enhancement**: Adjusts the contrast to make text more legible.
- **Brightness Enhancement**: Increases the brightness for darker images.
- **Adaptive Threshold**: Uses adaptive thresholding for dynamic text enhancement.
- **Deskewing**: Straightens tilted text images.

**Selection Process**: The first page of the document undergoes all enhancement methods with Tesseract OCR, and the best-performing one is selected automatically based on confidence score. Users can also override this by specifying their desired enhancement function.

---

### 2. **OCR Comparisons**

The package performs OCR with GPT models and compares the output with:

1. **Tesseract OCR**: A traditional OCR engine.
2. **GPT Base OCR**: A variant of GPT that avoids guessing ambiguous text.

#### Metrics for Comparison

The following metrics are used to compare the text outputs:

- **Similarity Ratio**: Measures text similarity as a percentage.
- **Normalized Levenshtein Distance**: Evaluates edit distances, normalized for text length differences.
- **Jaccard Similarity**: Compares sets of unique words to identify overlap.
- **Cosine Similarity**: Measures vector similarity based on text embeddings.
- **Perplexity**: Assesses how well GPT predicts the text output.
- **Average Probability (GPT)**: The average confidence of GPT for the generated text. **The primary indicator of GPT-OCR quality.**
- **Average Confidence (Tesseract)**: The mean confidence score provided by Tesseract OCR. **The primary indicator of Tesseract quality.**

---

### 3. **Token Anomaly Detection**

The package identifies tokens with the lowest log probabilities (e.g., `token_outlier=-2` for tokens in the bottom two log-probability ranges). This feature helps highlight anomalies or inconsistencies in the text that may require further review.

---

### 4. **Cost Estimation**

Since GPT-based OCR involves API usage, the package provides a detailed cost breakdown:

- **Cost Per Page**: The estimated cost for processing each page using OCR-GPT.
- **Total Cost**: Sum of all processing costs, including comparisons with GPT Base and Tesseract.

Example Cost Breakdown:
```
Total cost: $15.00 = OCR_GPT($10.00) + GPT_Base($3.00) + GPT_Test($2.00)
```


### 5. **Additional Features**

#### Batch Processing

The `implement_ocr_batch()` function allows processing multiple files in a batch, making it suitable for large-scale OCR tasks.

#### Folder Processing

The `implement_ocr_folder()` function allows processing multiple files in a given folder.

#### Page Numbering

Enabling `page_num=True` appends page numbers to the output text file for easier reference.

#### Estimated Cost Per Page

The estimated cost per page and total cost are displayed, allowing users to budget GPT API usage effectively.

---

## Function Descriptions

### `implement_ocr()`

Processes the PDF and returns OCR results alongside metrics.

#### Parameters:

- **`filename`**: Path to the PDF file.
- **`compare_gpt`**: Enables OCR comparison (`True`/`False`).
- **`page_num`**: Adds page numbers to the text (`True`/`False`).
- **`api_key`**: OpenAI API key for OCR-GPT.
- **`language`**: Language of a document (default: `English`).
- **`token_outlier`**: Specifies the anomaly detection threshold (default: `-2`).
- **`enhance`**: Enables image enhancement (default: `False`).
- **`cost_per_1000_tokens`**: Cost per 1000 tokens (default: `0.03`).

#### Returns:
A dictionary with the following keys:

- **`gpt_ocr_text`**: The combined text from all pages.
- **`metrics`**: A DataFrame containing comparison metrics.
- **`lowest_logprob_tokens`**: Tokens with low confidence for anomaly detection.
- **`gpt_test`**: Results of test queries on the OCR-ed text.
- **`total_cost`**: Estimated cost summary.
---

## Technical Workflow

1. **Image Conversion**: PDF pages are converted to images at a DPI of 300.
2. **Image Enhancement**: If enabled, the best enhancement method is selected based on the first page.
3. **OCR Execution**:
   - GPT performs OCR using a predefined prompt.
   - Optionally, GPT Base or Tesseract runs OCR for comparison.
4. **Metric Calculation**: Metrics like similarity ratio and perplexity are calculated for all comparisons.
5. **Anomaly Detection**: Tokens with low log probabilities are extracted for review.
6. **Cost Estimation**: Costs are calculated for each page and aggregated.
7. **Output Generation**:
   - The enhanced OCR text is saved as a file.
   - Metrics and additional data are returned in a structured format.

---

### `implement_ocr_batch()`

Processes a list of PDF files and aggregates OCR results and metrics for each file.

#### Parameters:

- **`file_list`**: List of file paths to the PDFs to process.
- **`kwargs`**: Additional arguments to customize OCR behavior, passed to `implement_ocr()`.

#### Function Workflow:

1. Iterates through each file in the provided list.
2. Extracts the file name using `os.path.basename`.
3. Calls the `implement_ocr()` function to perform OCR and retrieve metrics.
4. Processes the metrics to compute averages, excluding the last element.
5. Rounds each average to three decimal places and assigns each as a column.
6. Extracts the total cost using a regex from the OCR output and appends it to the results.

#### Returns:
- processed PDF files in txt format.
- **`summary_df`**: A DataFrame containing the following columns:
  - **`filename`**: Name of the processed file.
  - **`Est.Total Cost`**: Total estimated cost for OCR processing.
  - **`Sim GPT-Tess`**: Similarity metric between GPT and Tesseract OCR outputs.
  - **`Lev GPT-Tess`**: Levenshtein distance-based metric.
  - **`Jac GPT-Tess`**: Jaccard similarity metric.
  - **`CosSim GPT-Tess`**: Cosine similarity metric.
  - **`SimR GPT-Base`**, **`Lev GPT-Base`**, **`Jac GPT-Base`**, **`CosSim GPT-Base`**: Similarity metrics for GPT and base models.
  - **`Perp GPT`**, **`Perp Base`**: Perplexity values for GPT and base models.
  - **`Av.Prob.GPT`**, **`Av.Conf.Tess`**, **`Av.Prob.Base`**: Average probabilities and confidences for respective models.

#### Notes:

- If the number of DataFrame columns does not match the expected column names, a warning is issued.

---

### `implement_ocr_folder()`

Processes all PDF files in a specified folder and summarizes the results.

#### Parameters:

- **`folder_path`**: Path to the folder containing the PDF files.
- **`kwargs`**: Additional arguments passed to `implement_ocr_batch()` for OCR customization.

#### Function Workflow:

1. Verifies that the provided `folder_path` is a valid directory.
2. Identifies all `.pdf` files in the directory and creates a `file_list` with their full paths.
3. If no PDF files are found, raises a `ValueError`.
4. Calls `implement_ocr_batch()` with the list of files and additional arguments.

#### Returns:

- processed PDF files in txt format.
- **`summary_df`**: A DataFrame summarizing the OCR results for all PDF files in the folder. Columns match those produced by `implement_ocr_batch()`.

#### Notes:
- Raises an error if the provided path is not a directory or contains no `.pdf` files.


Here's the structured function description based on your earlier template format:  

---

### `tesseract_gpt_ocr`  

**Description:**  
This function performs Optical Character Recognition (OCR) on a given file using Tesseract and optionally compares the extracted text with GPT-generated text for verification. It also provides cost estimation and text enhancement options.  

**Parameters:**  
- `filename` (*str*): The path to the image or PDF file for OCR processing.  
- `compare_gpt` (*bool, default=False*): If `True`, compares Tesseract-extracted text with GPT-generated text.  
- `page_num` (*bool, default=True*): If `True`, includes page numbers in the output.  
- `api_key` (*str, optional*): API key for GPT processing if `compare_gpt` is enabled.  
- `language` (*str, default="English"*): Language to be used for OCR processing.  
- `token_outlier` (*int, default=-2*): Threshold for detecting outliers in token distribution.  
- `cost_per_1000_tokens` (*float, default=0.015*): Estimated cost per 1,000 GPT tokens if `compare_gpt` is enabled.  
- `gpt_test` (*bool, default=False*): If `True`, runs a GPT test alongside Tesseract processing.  
- `enhance` (*bool, default=False*): If `True`, applies preprocessing enhancements to improve OCR accuracy.  
- `threshold_score_tesseract` (*float, default=0.90*): Confidence threshold for accepting Tesseract-extracted text.  

**Returns:**  
- Extracted text from the document, with optional GPT comparison and cost estimation.  

---

### `tesseract_gpt_ocr_list`  

**Description:**  
Processes multiple files for OCR using `tesseract_gpt_ocr`.  

**Parameters:**  
- `file_list` (*list of str*): A list of file paths to be processed.  
- `**kwargs`: Additional arguments to be passed to `tesseract_gpt_ocr`.  

**Returns:**  
- A dictionary mapping file names to their extracted text results.  

---

### `tesseract_gpt_ocr_folder`  

**Description:**  
Processes all image or PDF files in a specified folder using OCR.  

**Parameters:**  
- `folder_path` (*str*): Path to the folder containing files for OCR processing.  
- `**kwargs`: Additional arguments to be passed to `tesseract_gpt_ocr`.  

**Returns:**  
- A dictionary mapping file names to their extracted text results.  

---

### Example Outputs

1. **`implement_ocr_batch()` Output Example**:

| filename | Est.Total Cost | Sim GPT-Tess | Lev GPT-Tess | Jac GPT-Tess | CosSim GPT-Tess | SimR GPT-Base | Lev GPT-Base | Jac GPT-Base | CosSim GPT-Base | Perp GPT | Perp Base | Av.Prob.GPT | Av.Conf.Tess | Av.Prob.Base |
|----------|----------------|--------------|--------------|--------------|------------------|---------------|--------------|--------------|----------------|----------|----------|-------------|--------------|--------------|
| file1.pdf| 0.37           | 0.245        | 0.756        | 0.812        | 0.913            | 0.128         | 0.654        | 0.765        | 0.876          | 2.34    | 2.98    | 0.876       | 0.754        | 0.643        |
| file2.pdf| 0.15           | 0.250        | 0.801        | 0.850        | 0.950            | 0.15         | 0.670        | 0.800        | 0.885          | 2.34    | 3.98    | 0.895       | 0.735        | 0.834        |
| Total.Cost.All| 0.52           | none        | none        | none        | none            | none         | none        | none        | none          | none    | none    | none       | none        | none        |

2. **`implement_ocr_folder()` Output Example**:
   Produces a similar DataFrame as `implement_ocr_batch()` but aggregates results from all PDFs in the specified folder.
   

### 5. **Running OCR-GPT in Python**

Follow these steps to set up and run the OCR-GPT script in Python:

#### 1. **Download and Install Anaconda**

   - Visit [Anaconda's official website](https://www.anaconda.com/products/distribution) to download and install Anaconda.

#### 2. **Install Tesseract**

   - Download the Tesseract executable from [this link](https://github.com/UB-Mannheim/tesseract/wiki).
   
   - Install the Tesseract executable in the following directory:
   
   ```bash
   C:\Program Files (x86)\Tesseract-OCR
   ```

#### 3. **Run Anaconda Command Prompt**

   - Open the Anaconda Command Prompt (or Terminal in macOS/Linux).

#### 4. **Create Virtual Environment**

   - Create a new virtual environment named `"ocr_gpt"` (or any other name) by running the following command:
   
   ```bash
   conda create -n "ocr_gpt"
   ```

#### 5. **Download Packages**

   - Accept the prompts to download the required packages:
   
   ```bash
   y
   ```

#### 6. **Activate Virtual Environment**

   - Activate the virtual environment with the following command:
   
   ```bash
   conda activate "ocr_gpt"
   ```

#### 7. **Change Directory to OCR-GPT Script Location**

   - Change the directory to where the OCR-GPT script files are located. For example:
   
   ```bash
   cd C:\Users\PATH_TO_DIRECTORY
   ```

#### 8. **Install Required Packages**

   - Install pip if it is not already installed in your environment:
   
   ```bash
   conda install pip
   ```

   - If you are setting up a completely new environment and need all the packages to be installed, use the following command:
   
   ```bash
   pip install -r requirements.txt
   ```

   - If you are updating the OCR-GPT script and need only missing packages (e.g., updating from an older version), use this command:
   
   ```bash
   pip install --ignore-installed -r requirements.txt
   ```

   - **Note**: Some packages with C/C++ dependencies may require you to install a C++ compiler. Follow Python's instructions on how to download and install it.


#### 9. **Test Tesseract Installation**

   - To test if Tesseract is installed correctly, open a Python prompt and run the following commands:
   
   ```python
   import pytesseract
   print(pytesseract.get_tesseract_version())
   ```

#### 10. **Use OCR-GPT**

   - After setting up, you can start using OCR-GPT for optical character recognition (OCR) and text processing with GPT-3. Here's an example of how to use it:

   ```python
   import ocr_gpt as ocr
   import openai
   import os

   # Set up OpenAI API key
   openai.api_key = os.environ["OPENAI_API_KEY"]
   api_key = os.environ["OPENAI_API_KEY"]

   # Single file OCR
   filename = "C:/Users/Desktop/OCR/file1.pdf"
   result1 = ocr.implement_ocr(filename, api_key=api_key, 
                                  compare_gpt=True, gpt_test=True,
                                  token_outlier=-2, page_num=False, enhance=False)

   # Process a batch of files
   file_list = ["file1.pdf", "file2.pdf"]
   result2 = ocr.implement_ocr_list(file_list, api_key=api_key, 
                                    compare_gpt=True, page_num=True,
                                    gpt_test=False)

   # Process all PDF files in a folder
   folder_path = "C:/Users/Desktop/OCR"
   result3 = ocr.implement_ocr_folder(folder_path, api_key=api_key, enhance=True)
   
   
   # Process all PDF files in a list using Tesseract or OCR
   folder_path = "C:/Users/Desktop/OCR"
   result4 = ocr.tesseract_gpt_ocr_list(file_list, 
                                        api_key=api_key, enhance=False,
                                        threshold_score_tesseract=0.90)
   
   ```
   
### 6. **Command-Line Interface (CLI) Support**

The package can be executed directly from the command line for easy integration into workflows. 

**Example Command:**
```bash
python ocr_package.py --filename sample.pdf --api_key <your_openai_api_key> --compare_gpt True --page_num True --enhance True --token_outlier -2
```

**Arguments:**
- `--filename`: Path to the input PDF file.
- `--api_key`: Your OpenAI API key.
- `--compare_gpt`: Enables comparison mode (default: `False`).
- `--page_num`: Adds page numbers to the output text (default: `True`).
- `--enhance`: Enables image enhancement (default: `False`).
- `--token_outlier`: Adjusts sensitivity for anomaly detection (default: `-2`).

**Output Example:**
Upon completion, the command will output:
- Enhanced OCR text saved to `<filename>_ocrgpt.txt`.
- Metrics and logs displayed in the terminal and saved to a JSON or CSV file.


Here are examples of how to run the script from the command line using all three modes (`single`, `batch`, and `folder`). Replace `script_name.py` with the actual name of your Python script.

---

### **1. Single File Processing**
```bash
python ocr_gpt.py --mode single --filename sample.pdf --compare_gpt true --page_num false --api_key your_openai_api_key --token_outlier -1 --gpt_test true --enhance false
```

### **2. Batch Processing**
```bash
python ocr_gpt.py --mode batch --file_list file1.pdf file2.pdf file3.pdf --compare_gpt true --page_num true --api_key your_openai_api_key --token_outlier -2 --gpt_test false --enhance true
```

### **3. Folder Processing**
```bash
python ocr_gpt.py --mode folder --folder_path ./pdf_folder --compare_gpt false --page_num true --api_key your_openai_api_key --token_outlier -3 --gpt_test true --enhance true
```

---

### **4. Single File Processing with Default Options**
If you want to use mostly default values:
```bash
python ocr_gpt.py --mode single --filename document.pdf
```

---

### **5. Batch Processing with Minimal Parameters**
Only specify mandatory arguments:
```bash
python ocr_gpt.py --mode batch --file_list file1.pdf file2.pdf
```

---

### Explanation of Parameters
- `--mode`: Required argument to select the operation mode (`single`, `batch`, or `folder`).
- `--filename`: Path to the file for single processing mode.
- `--file_list`: List of files for batch processing mode.
- `--folder_path`: Path to a folder containing files for folder processing mode.
- `--compare_gpt`: Whether to compare GPT-based OCR with another method (`true`/`false`).
- `--page_num`: Whether to include page numbers in the output (`true`/`false`).
- `--api_key`: OpenAI API key for GPT OCR.
- `--token_outlier`: Threshold for outlier token handling.
- `--gpt_test`: Whether to run additional GPT tests (`true`/`false`).
- `--enhance`: Whether to apply image enhancement before OCR (`true`/`false`).


This package provides robust OCR capabilities by integrating GPT with traditional OCR tools and enhancement features, ensuring high accuracy and transparency in results.
