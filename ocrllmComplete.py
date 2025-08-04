# ocrllmComplete.py
# This script performs OCR on a PDF file, processes the text using GPT-4 or DeepSeek, and saves the corrected output.
# It uses Tesseract or Google Vision for OCR and OpenAI's GPT-4 or DeepSeek hosted on OpenRouter for text correction.
# Comment and uncomment the relevant function calls to switch between methods.


# dependencies
import os
import time
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import openai
import traceback
from google.cloud import vision
import glob
import requests

# file paths
TEMP_IMG_FOLDER = "pdf_pages"
FIRST_OUTPUT_FILE = "tesseract.txt" # or vision.txt
CORRECTED_OUTPUT_FILE = "gptTesseract.txt" # or any other combination of ocr agent and LLM
PDF_FILE = "your_pdf_file.pdf"  # replace with your actual PDF file path

# API keys
OPENAI_API_KEY = 'YOUR_GPT_API_KEY'
openai.api_key = OPENAI_API_KEY
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "YOUR_GOOGLE_CREDENTIALS_JSON_PATH"
client = vision.ImageAnnotatorClient()
API_KEY = "YOUR_OPENROUTER_KEY"

# batch processing constants
MAX_RETRIES_GPT4 = 3
MAX_CHARS_PER_BATCH = 1500
MODEL = "deepseek/deepseek-r1-0528:free"

# PDF to image using the pdf2image library, saves images to TEMP_IMG_FOLDER
def pdf_to_images(pdf_path, output_folder):
    print("[INFO] Converting PDF to images")
    os.makedirs(output_folder, exist_ok=True)
    pages = convert_from_path(pdf_path)
    print(f"[DEBUG] Pages converted: {len(pages)}")
    image_paths = []

    for i, page in enumerate(pages):
        img_path = os.path.join(output_folder, f"page_{i+1:03d}.png")
        page.save(img_path, "PNG")
        image_paths.append(img_path)
    print(f"[INFO] PDF converted to {len(image_paths)} images.")
    return image_paths

# OCR using Tesseract
def run_tesseract(image_paths):
    print("[INFO] Running OCR on images...")
    output_lines = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img, lang="san+hin")  # no option for Pali, so this should suffice
            output_lines.append(text.strip())
        except Exception as e:
            print(f"[ERROR] OCR failed on {img_path}: {e}")

    with open(FIRST_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(output_lines))

    print(f"[INFO] OCR complete. Output saved to {FIRST_OUTPUT_FILE}")

# OCR using Google Vision
def run_vision(image_paths):
    def extract_text_from_image(image_path):
        """Extracts full text from a single image using Google Vision."""
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        return response.full_text_annotation.text if response.full_text_annotation.text else ""

    with open(FIRST_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, path in enumerate(image_paths):
            print(f"OCR processing page {i+1}/{len(image_paths)}: {path}")
            text = extract_text_from_image(path)
            f.write(text + "\n\n")

    print(f"✅ OCR completed. Output saved to {FIRST_OUTPUT_FILE}")

# GPT-4 text correction
def gpt_parse():
    with open(FIRST_OUTPUT_FILE, "r", encoding="utf-8") as f:
        raw_ocr_text = f.read()

    # Simple paragraph split batching
    paragraphs_gpt4 = raw_ocr_text.split("\n\n")
    openai_batches = []
    buffer = ""

    for para in paragraphs_gpt4:
        if len(buffer) + len(para) < MAX_CHARS_PER_BATCH:
            buffer += para + "\n\n"
        else:
            openai_batches.append(buffer.strip())
            buffer = para + "\n\n"
    if buffer:
        openai_batches.append(buffer.strip())

    print(f"[GPT-4] Prepared {len(openai_batches)} batches.")

    with open(CORRECTED_OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for i, batch in enumerate(openai_batches):
            if not batch.strip():
                print(f"[SKIP] Batch {i+1} is empty.")
                continue

            print(f"[GPT-4] Sending batch {i+1}/{len(openai_batches)}...")

            for attempt in range(1, MAX_RETRIES_GPT4 + 1):
                try:
                    start = time.time()

                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": f"""You are supposed to check the following text for spelling errors in Pali, Sanskrit and Hindi. Return ONLY the formatted text, not a single word of English or any other lamguage. All of this will go into a text file for direct evaluation with CER. No "Here's the text" or anything like that. Look carefully and give ONLY, I repeat ONLY the Devanagari text. Keep the page and line breaks as they are.\n\n{batch}"""
                            }
                        ],
                        temperature=0.2,
                        timeout=30,
                    )

                    corrected = response.choices[0].message.content.strip()
                    out_file.write(corrected + "\n\n")
                    out_file.flush()

                    print(f"[GPT-4] Batch {i+1} done in {time.time() - start:.2f}s")
                    time.sleep(1)
                    break  # success

                except Exception as e:
                    print(f"[GPT-4] Error on batch {i+1}, attempt {attempt}")
                    traceback.print_exc()
                    if attempt == MAX_RETRIES_GPT4:
                        print(f"[GPT-4] Skipping batch {i+1}")
                    else:
                        print(f"[GPT-4] Retrying batch {i+1}...")
                        time.sleep(2)

# DeepSeek text correction
def deepseek_parse():
    with open(FIRST_OUTPUT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    paragraphs = raw_text.split("\n\n")
    batches = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) < MAX_CHARS_PER_BATCH:
            buffer += para + "\n\n"
        else:
            batches.append(buffer.strip())
            buffer = para + "\n\n"
    if buffer:
        batches.append(buffer.strip())
    
    print(f"[DEBUG] Total batches created: {len(batches)}")
    headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
    }

    with open(CORRECTED_OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for i, batch in enumerate(batches):
            prompt = f"""You are supposed to check the following text for spelling errors in Pali, Sanskrit and Hindi. Return ONLY the text, not a single word of English or any other lamguage. All of this will go into a text file for direct evaluation with CER. No "Here's the text" or anything like that. Look carefully and give ONLY, I repeat ONLY the Devanagari text. Do not hallucinate.\n\n{batch}"""
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}]
            }

            print(f"[INFO] Sending batch {i+1}/{len(batches)}...")
            try:
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                corrected = response.json()["choices"][0]["message"]["content"]
                out_f.write(corrected + "\n\n")
                time.sleep(1.5)
            except Exception as e:
                print(f"[ERROR] Batch {i+1} failed: {e}")


if __name__ == "__main__":
    if not os.path.exists(PDF_FILE):
        print(f"[ERROR] PDF file '{PDF_FILE}' not found.")
    else:
        image_paths = pdf_to_images(PDF_FILE, TEMP_IMG_FOLDER)
        run_tesseract(image_paths) # or run_vision(image_paths)
        gpt_parse() # or deepseek_parse()
        print("[INFO] Processing complete.")
