import os
import cv2
import pytesseract
import numpy as np
from PIL import Image
import requests
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_202_ACCEPTED
import io
import fitz  # PyMuPDF
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import time
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Contract Analyzer API",
    description="API for analyzing startup contracts with focus on NDAs and SaaS agreements",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants and configuration
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    logger.error("PERPLEXITY_API_KEY not set in environment variables. Please check your .env file.")
    raise ValueError("PERPLEXITY_API_KEY environment variable is required")
logger.info(f"Loaded Perplexity API Key: {PERPLEXITY_API_KEY[:4]}...{PERPLEXITY_API_KEY[-4:]}")

TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "/opt/homebrew/bin/tesseract")
logger.info(f"Using Tesseract command path: {TESSERACT_CMD}")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Storage for asynchronous job results
job_results = {}

# Pydantic models for request and response validation
class ContractAnalysisResponse(BaseModel):
    summary: str
    contract_type: str
    risks: List[Dict[str, str]]
    suggestions: List[Dict[str, str]]
    missing_clauses: List[Dict[str, str]]
    key_terms: List[Dict[str, Any]]

class ContractType(BaseModel):
    contract_type: str

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Contract Analyzer API is running"}

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """
    Extract text from a PDF file using PyMuPDF and Tesseract OCR
    """
    try:
        logger.info("Processing PDF file")
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        extracted_text = ""
        
        if len(doc) == 0:
            logger.error("PDF document appears to be empty or invalid")
            raise ValueError("PDF document is empty or invalid")
        
        for page_num in range(len(doc)):
            logger.info(f"Processing page {page_num+1}/{len(doc)}")
            page = doc.load_page(page_num)
            text = page.get_text()
            
            if not text.strip():
                logger.info(f"Page {page_num+1} appears to be image-based, using OCR")
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    open_cv_image = np.array(img)
                    open_cv_image = open_cv_image[:, :, ::-1].copy()
                    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    text = pytesseract.image_to_string(gray)
                except Exception as ocr_error:
                    logger.error(f"OCR processing error on page {page_num+1}: {str(ocr_error)}")
                    text = f"[OCR PROCESSING ERROR ON PAGE {page_num+1}]"
            
            extracted_text += text + "\n\n"
        
        if not extracted_text.strip():
            logger.warning("No text was extracted from the PDF")
            return "No text could be extracted from the document."
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters from the PDF")
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

def analyze_contract_with_perplexity(contract_text: str, contract_type: str) -> Dict:
    """
    Analyze contract text using Perplexity Sonar API
    """
    try:
        logger.info(f"Sending contract text to Perplexity API (type: {contract_type})")
        if not PERPLEXITY_API_KEY:
            logger.error("Perplexity API key is missing")
            raise ValueError("Valid Perplexity API key is required")
        
        max_text_length = 15000
        if len(contract_text) > max_text_length:
            logger.info(f"Contract text too long ({len(contract_text)} chars), truncating to {max_text_length}")
            contract_text = contract_text[:max_text_length]
        
        if contract_type.lower() == "nda":
            prompt = f"""
            Analyze this Non-Disclosure Agreement (NDA) contract:
            
            {contract_text}
            
            Provide a detailed analysis in JSON format with the following sections:
            1. summary: A brief summary of the NDA contract (max 150 words)
            2. risks: List of potential risks for a startup in this contract, with 'description' and 'severity' (high/medium/low) for each
            3. suggestions: List of suggestions to improve the contract with 'description' and 'importance' (high/medium/low) for each
            4. missing_clauses: Standard NDA clauses that are missing from this contract with 'clause_name' and 'description' for each
            5. key_terms: List of important terms, each as a dictionary with 'term' and 'value' keys, including confidentiality period, jurisdiction, remedies, and obligations
            
            Provide JSON response without any additional text or explanations.
            """
        elif contract_type.lower() == "saas":
            prompt = f"""
            Analyze this Software-as-a-Service (SaaS) contract:
            
            {contract_text}
            
            Provide a detailed analysis in JSON format with the following sections:
            1. summary: A brief summary of the SaaS contract (max 150 words)
            2. risks: List of potential risks for a startup in this contract, with 'description' and 'severity' (high/medium/low) for each
            3. suggestions: List of suggestions to improve the contract with 'description' and 'importance' (high/medium/low) for each
            4. missing_clauses: Standard SaaS clauses that are missing from this contract with 'clause_name' and 'description' for each
            5. key_terms: List of important terms, each as a dictionary with 'term' and 'value' keys, including payment terms, SLAs, data ownership, liability limits, and termination conditions
            
            Provide JSON response without any additional text or explanations.
            """
        else:
            prompt = f"""
            Analyze this contract:
            
            {contract_text}
            
            First determine what type of contract this is. Then provide a detailed analysis in JSON format with the following sections:
            1. summary: A brief summary of the contract (max 150 words)
            2. contract_type: The identified type of contract
            3. risks: List of potential risks for a startup in this contract, with 'description' and 'severity' (high/medium/low) for each
            4. suggestions: List of suggestions to improve the contract with 'description' and 'importance' (high/medium/low) for each
            5. missing_clauses: Standard clauses that are missing from this contract with 'clause_name' and 'description' for each
            6. key_terms: List of important terms, each as a dictionary with 'term' and 'value' keys, including payment, termination, liability, and other relevant terms
            
            Provide JSON response without any additional text or explanations.
            """
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 4000
        }
        
        logger.debug(f"Sending request to Perplexity API with headers: {headers}")
        logger.debug(f"Request body: {json.dumps(data, indent=2)}")
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    break
                elif response.status_code == 401:
                    logger.error(f"Authorization error (401) from Perplexity API: {response.text}")
                    raise HTTPException(
                        status_code=500,
                        detail="Perplexity API returned 401 Unauthorized. Please check your API key in the .env file, ensure sufficient account credits, or contact api@perplexity.ai."
                    )
                elif response.status_code == 429:
                    logger.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}), retrying in {retry_delay} seconds")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Error from Perplexity API: {response.status_code}, {response.text}")
                    if attempt == max_retries - 1:
                        raise HTTPException(status_code=500, detail=f"Error from Perplexity API: {response.text}")
                    time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=500, detail=f"Request to Perplexity API failed: {str(e)}")
                time.sleep(retry_delay)
        
        result = response.json()
        logger.debug(f"Perplexity API response: {json.dumps(result, indent=2)}")
        
        if 'choices' not in result or not result['choices']:
            logger.error(f"Invalid response from Perplexity API: {result}")
            raise HTTPException(status_code=500, detail="Invalid response from Perplexity API")
        
        analysis_text = result['choices'][0]['message']['content']
        analysis = extract_json_from_text(analysis_text)
        
        # Transform key_terms to a list of dictionaries if it's a single dictionary
        if "key_terms" in analysis and isinstance(analysis["key_terms"], dict):
            key_terms_dict = analysis["key_terms"]
            analysis["key_terms"] = [
                {"term": key, "value": value} for key, value in key_terms_dict.items()
            ]
        
        required_fields = ["summary", "risks", "suggestions", "missing_clauses", "key_terms"]
        for field in required_fields:
            if field not in analysis:
                analysis[field] = []
                if field == "summary":
                    analysis[field] = "Analysis could not extract a proper summary."
        
        if "contract_type" not in analysis:
            analysis["contract_type"] = contract_type
        
        logger.info("Contract analysis completed successfully")
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing contract with Perplexity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing contract: {str(e)}")

def extract_json_from_text(text: str) -> Dict:
    """
    Extract JSON from text response which might have additional text
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_pattern = r'({[\s\S]*})|$([\s\S]*)$'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(0)
                json_str = re.sub(r'```(?:json)?\s*|\s*```', '', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse extracted JSON pattern: {str(e)}")
        
        lines = text.split('\n')
        json_start = -1
        json_end = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('{') and json_start == -1:
                json_start = i
            if line.strip().endswith('}') and json_start != -1:
                json_end = i
                break
        
        if json_start != -1 and json_end != -1:
            json_text = '\n'.join(lines[json_start:json_end+1])
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass
        
        logger.warning("Could not extract JSON from Perplexity response")
        return {
            "summary": text[:500],
            "risks": [],
            "suggestions": [],
            "missing_clauses": [],
            "key_terms": []
        }

def async_process_contract(job_id: str, contents: bytes, contract_type: str):
    """
    Process contract asynchronously in the background
    """
    try:
        job_results[job_id] = {"status": "processing"}
        extracted_text = extract_text_from_pdf(contents)
        
        if not contract_type:
            detection_text = extracted_text[:10000]
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "sonar",
                "messages": [{"role": "user", "content": f"Based on the following contract text, determine if this is an NDA, SaaS agreement, or another type of contract. Respond with only the contract type (e.g., 'NDA', 'SaaS', or 'Other').\n\n{detection_text}"}],
                "temperature": 0.1,
                "max_tokens": 50
            }
            logger.debug(f"Sending contract type detection request with headers: {headers}")
            logger.debug(f"Request body: {json.dumps(data, indent=2)}")
            try:
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=15
                )
                if response.status_code == 200:
                    result = response.json()
                    detected_contract_type = result['choices'][0]['message']['content'].strip()
                    if "nda" in detected_contract_type.lower() or "non-disclosure" in detected_contract_type.lower():
                        contract_type = "NDA"
                    elif "saas" in detected_contract_type.lower() or "software as a service" in detected_contract_type.lower():
                        contract_type = "SaaS"
                    else:
                        contract_type = "Other"
                    logger.info(f"Auto-detected contract type: {contract_type}")
                elif response.status_code == 401:
                    logger.error(f"Authorization error (401) from Perplexity API: {response.text}")
                    raise HTTPException(
                        status_code=500,
                        detail="Perplexity API returned 401 Unauthorized during contract type detection. Please check your API key in the .env file, ensure sufficient account credits, or contact api@perplexity.ai."
                    )
                else:
                    contract_type = "Other"
                    logger.warning(f"Contract type detection failed with status {response.status_code}, defaulting to 'Other'")
            except Exception as detect_error:
                contract_type = "Other"
                logger.warning(f"Contract type detection error: {str(detect_error)}, defaulting to 'Other'")
        
        analysis = analyze_contract_with_perplexity(extracted_text, contract_type)
        job_results[job_id] = {
            "status": "completed",
            "result": analysis,
            "completion_time": time.time()
        }
    except Exception as e:
        logger.error(f"Error in async processing job {job_id}: {str(e)}")
        job_results[job_id] = {
            "status": "failed",
            "error": str(e),
            "completion_time": time.time()
        }

@app.post("/detect-contract-type", response_model=ContractType)
async def detect_contract_type(file: UploadFile = File(...)):
    """
    Detect the type of contract from an uploaded PDF
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        logger.info(f"Detecting contract type for file: {file.filename}")
        contents = await file.read()
        
        extracted_text = extract_text_from_pdf(contents)
        detection_text = extracted_text[:10000]
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "sonar",
            "messages": [{"role": "user", "content": f"Based on the following contract text, determine if this is an NDA, SaaS agreement, or another type of contract. Respond with only the contract type (e.g., 'NDA', 'SaaS', or 'Other').\n\n{detection_text}"}],
            "temperature": 0.1,
            "max_tokens": 50
        }
        logger.debug(f"Sending contract type detection request with headers: {headers}")
        logger.debug(f"Request body: {json.dumps(data, indent=2)}")
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data,
                timeout=15
            )
            if response.status_code == 200:
                result = response.json()
                contract_type = result['choices'][0]['message']['content'].strip()
                
                if "nda" in contract_type.lower() or "non-disclosure" in contract_type.lower() or "confidentiality" in contract_type.lower():
                    contract_type = "NDA"
                elif "saas" in contract_type.lower() or "software as a service" in contract_type.lower() or "service agreement" in contract_type.lower():
                    contract_type = "SaaS"
                else:
                    contract_type = "Other"
                
                logger.info(f"Detected contract type: {contract_type}")
                return {"contract_type": contract_type}
            elif response.status_code == 401:
                logger.error(f"Authorization error (401) from Perplexity API: {response.text}")
                raise HTTPException(
                    status_code=500,
                    detail="Perplexity API returned 401 Unauthorized. Please check your API key in the .env file, ensure sufficient account credits, or contact api@perplexity.ai."
                )
            else:
                logger.error(f"Error from Perplexity API: {response.status_code}, {response.text}")
                raise HTTPException(status_code=500, detail="Error detecting contract type")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Perplexity API failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Request to Perplexity API failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error detecting contract type: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting contract type: {str(e)}")

@app.post("/analyze-contract", response_model=ContractAnalysisResponse)
async def analyze_contract(file: UploadFile = File(...), contract_type: str = Form(None)):
    """
    Analyze a contract uploaded as PDF file
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        logger.info(f"Analyzing contract: {file.filename}, Type: {contract_type}")
        contents = await file.read()
        
        extracted_text = extract_text_from_pdf(contents)
        
        if not contract_type:
            detection_text = extracted_text[:10000]
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "sonar",
                "messages": [{"role": "user", "content": f"Based on the following contract text, determine if this is an NDA, SaaS agreement, or another type of contract. Respond with only the contract type (e.g., 'NDA', 'SaaS', or 'Other').\n\n{detection_text}"}],
                "temperature": 0.1,
                "max_tokens": 50
            }
            logger.debug(f"Sending contract type detection request with headers: {headers}")
            logger.debug(f"Request body: {json.dumps(data, indent=2)}")
            try:
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=15
                )
                if response.status_code == 200:
                    result = response.json()
                    detected_contract_type = result['choices'][0]['message']['content'].strip()
                    if "nda" in detected_contract_type.lower() or "non-disclosure" in detected_contract_type.lower():
                        contract_type = "NDA"
                    elif "saas" in detected_contract_type.lower() or "software as a service" in detected_contract_type.lower():
                        contract_type = "SaaS"
                    else:
                        contract_type = "Other"
                    logger.info(f"Auto-detected contract type: {contract_type}")
                elif response.status_code == 401:
                    logger.error(f"Authorization error (401) from Perplexity API: {response.text}")
                    raise HTTPException(
                        status_code=500,
                        detail="Perplexity API returned 401 Unauthorized during contract type detection. Please check your API key in the .env file, ensure sufficient account credits, or contact api@perplexity.ai."
                    )
                else:
                    contract_type = "Other"
                    logger.warning(f"Contract type detection failed with status {response.status_code}, defaulting to 'Other'")
            except Exception as detect_error:
                contract_type = "Other"
                logger.warning(f"Contract type detection error: {str(detect_error)}, defaulting to 'Other'")
        
        analysis = analyze_contract_with_perplexity(extracted_text, contract_type)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing contract: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing contract: {str(e)}")

@app.post("/analyze-contract-async", response_model=JobResponse)
async def analyze_contract_async(background_tasks: BackgroundTasks, file: UploadFile = File(...), contract_type: str = Form(None)):
    """
    Analyze a contract asynchronously (for large files)
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        job_id = f"job_{int(time.time())}_{file.filename}"
        logger.info(f"Starting async job {job_id} for contract: {file.filename}, Type: {contract_type}")
        contents = await file.read()
        
        job_results[job_id] = {"status": "queued"}
        background_tasks.add_task(async_process_contract, job_id, contents, contract_type)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Your contract is being processed. Use the /job-status endpoint to check progress."
        }
    except Exception as e:
        logger.error(f"Error starting async contract analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting contract analysis: {str(e)}")

@app.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Check the status of an asynchronous job
    """
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Job not found")
    job_data = job_results[job_id]
    
    current_time = time.time()
    if job_data.get("completion_time", current_time) < current_time - 3600 and job_data["status"] in ["completed", "failed"]:
        job_data["pending_cleanup"] = True
    
    response = {
        "job_id": job_id,
        "status": job_data["status"]
    }
    
    if job_data["status"] == "completed":
        response["result"] = job_data["result"]
    elif job_data["status"] == "failed":
        response["result"] = {"error": job_data.get("error", "Unknown error")}
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True) 