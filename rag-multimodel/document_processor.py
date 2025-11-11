"""
document_processor.py - Step 1 (Final, Structure-Aware): Intelligent Visual Extraction

Author: Ravi
Date: 2023-11-08

This module uses the PDF's internal structure (drawings and text blocks) to
intelligently identify, bound, and extract visual elements.
"""

import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import List

from logger import logger
from config import settings

def extract_visuals_from_pdf(file_path: str) -> List[str]:
    """
    Extracts visual elements by analyzing the PDF's structural components like
    drawings and text blocks to create perfect, dynamically-sized bounding boxes.
    """
    saved_image_paths = []
    try:
        doc = fitz.open(file_path)
        logger.info(f"Processing file for visuals: {file_path}")

        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 1. Get all drawing paths (vector graphics)
            paths = page.get_drawings()
            if not paths:
                continue

            # 2. Group nearby drawings into a single bounding box for the chart
            chart_bbox = fitz.Rect()
            for path in paths:
                chart_bbox.include_rect(path["rect"])

            if chart_bbox.is_empty or chart_bbox.width < 50 or chart_bbox.height < 50:
                continue

            # 3. Create a slightly larger "search area" around the chart
            search_area = fitz.Rect(chart_bbox)
            search_area.x0 -= 20
            search_area.y0 -= 20
            search_area.x1 += 20
            search_area.y1 += 20

            # 4. Find all text blocks within the search area to create the final bounding box
            final_bbox = fitz.Rect(chart_bbox) # Start with the chart's box
            for block in page.get_text("blocks"):
                block_rect = fitz.Rect(block[:4])
                # If the text block is inside our search area, include it
                if search_area.intersects(block_rect):
                    final_bbox.include_rect(block_rect)

            # 5. Render the page and perform the surgical crop
            # Add a final padding for safety
            padding = 15
            final_bbox.x0 -= padding
            final_bbox.y0 -= padding
            final_bbox.x1 += padding
            final_bbox.y1 += padding
            
            # Ensure the bounding box is within the page boundaries
            final_bbox.intersect(page.rect)

            if final_bbox.is_empty or final_bbox.width < 100 or final_bbox.height < 100:
                continue

            pix = page.get_pixmap(dpi=200, clip=final_bbox)
            page_image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if page_image.shape[-1] == 4:
                page_image = cv2.cvtColor(page_image, cv2.COLOR_BGRA2BGR)

            # 6. Save the final cropped image
            image_filename = f"{os.path.basename(file_path).replace('.pdf', '')}_p{page_num+1}_v{len(saved_image_paths)}.png"
            image_path = os.path.join(settings.image_output_folder, image_filename)
            cv2.imwrite(image_path, page_image)
            
            # Return the absolute path to the image
            absolute_image_path = os.path.abspath(image_path)
            saved_image_paths.append(absolute_image_path)
            logger.info(f"Saved visual element: {absolute_image_path}")

        doc.close()
    except Exception as e:
        logger.error(f"Could not process file {file_path}: {e}", exc_info=True)
    
    return saved_image_paths

def extract_text_from_pdf(file_path: str) -> List[dict]:
    """Extracts raw text from each page of a PDF."""
    text_data = []
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text:
                text_data.append({
                    "text": text,
                    "metadata": {"source": file_path, "page": page_num + 1}
                })
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}", exc_info=True)
    return text_data
