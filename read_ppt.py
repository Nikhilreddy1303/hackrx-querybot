from pptx import Presentation
import logging

def read_ppt_and_chunk(ppt_path: str):
    """
    Reads a PowerPoint file, extracting text from shapes on each slide.
    """
    try:
        prs = Presentation(ppt_path)
        for i, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text.append(shape.text)
            
            full_slide_text = "\n".join(slide_text).strip()
            if full_slide_text:
                yield {
                    "text": full_slide_text,
                    "source": f"Slide {i + 1}"
                }
    except Exception as e:
        logging.error(f"Could not process PowerPoint file: {e}")
        return