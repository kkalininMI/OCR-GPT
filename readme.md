---
title: "OCR GPT Guide"
author: "Kirill Kalinin"
date: "2024-11-30"
output: html_document
---

# Detailed Description of `ocr_gpt` Package

# GPT OCR Package Documentation

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
- **Average Probability (GPT)**: The average confidence of GPT for the generated text.
- **Average Confidence (Tesseract)**: The mean confidence score provided by Tesseract OCR.

---

### 3. **Token Anomaly Detection**

The package identifies tokens with the lowest log probabilities (e.g., `token_outlier=-2` for tokens in the bottom two log-probability ranges). This feature helps highlight anomalies or inconsistencies in the text that may require further review.

---

### 4. **Cost Estimation**

Since GPT-based OCR involves API usage, the package provides a detailed cost breakdown:

- **Cost Per Page**: The estimated cost for processing each page using GPT OCR.
- **Total Cost**: Sum of all processing costs, including comparisons with GPT Base and Tesseract.

Example Cost Breakdown:
```
Total cost: $15.00 = GPT_OCR($10.00) + GPT_Base($3.00) + GPT_Test($2.00)
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
- **`api_key`**: OpenAI API key for GPT OCR.
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
   



### 5. **Command-Line Interface (CLI) Support**

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
