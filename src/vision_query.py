import base64
from pathlib import Path

import numpy as np
from typing import Union
from openai import OpenAI

from config import BASE_DIR, IMG_FOLDER
from faiss_utils import load_faiss_index, normalize

def search_image_by_question(question, co, top_k=1):
    # Embed the question correctly
    response = co.embed(
        texts=[question],
        input_type="search_query",
        model="embed-v4.0"
    )
    query_emb = response.embeddings.float[0]  # ✅ Correct access

    index, filenames = load_faiss_index()
    norm_query = normalize(np.array(query_emb)).astype("float32")
    
    D, I = index.search(norm_query[np.newaxis, :], top_k)
    matched_paths = []
    for idx in I[0]:
        if idx < len(filenames):
            absolute_path = (IMG_FOLDER / filenames[idx]).resolve()
            try:
                relative_path = absolute_path.relative_to(BASE_DIR)
            except ValueError:
                relative_path = absolute_path
            matched_paths.append(str(relative_path))
    print("📂 matched_paths:", matched_paths)
    return matched_paths

def encode_image_to_base64(img_path: str) -> str:
    """Encodes an image to base64 for embedding in a prompt."""
    resolved_path = _resolve_path(img_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Image not found: {resolved_path}")
    with resolved_path.open("rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def _resolve_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (BASE_DIR / candidate)


def answer_question_about_images(question: str, matched_paths: list, client: OpenAI,
                                 model="gpt-4.1-mini", verbose=True) -> str:
    """
    Sends a multimodal prompt (text + multiple images) to the LLM and returns the answer.

    Parameters:
    - question (str): User query
    - matched_paths (list): List of local image paths
    - client: OpenAI or AzureOpenAI client
    - model (str): Model to use (e.g., gpt-4.1-mini, gpt-4o)
    - verbose (bool): Whether to print the response

    Returns:
    - response text
    """
    try:
        image_contents = []
        image_captions = []
        for img_path in matched_paths:
            b64 = encode_image_to_base64(img_path)
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

            resolved = _resolve_path(img_path)
            stem = resolved.stem
            if "_page" in stem:
                doc_slug, page_label = stem.split("_page", 1)
                page_label = page_label or "unknown"
            else:
                doc_slug, page_label = stem, "unknown"
            image_captions.append(
                f"- {resolved.name} (Document: {doc_slug}, page: {page_label})"
            )
        if not image_captions:
            image_captions.append("- No images retrieved.")

        context_prompt = (
            "You analyze scanned pages from World Bank Trust Fund annual reports. "
            "Use only the provided images to answer user questions. "
            "When possible, quote exact figures/text, name the chart or table you relied on, "
            "and keep responses concise (2–4 sentences or bullet points). "
            "If the images do not contain the answer, say so explicitly."
        )

        user_prompt = (
            "Question: {question}\n"
            "Instructions:\n"
            "- Inspect every attached image (each corresponds to a PDF page).\n"
            "- Summarize the relevant insight and cite the image filename, page label, and source document name.\n"
            "- Highlight quantitative values directly from the visuals.\n"
            "- If evidence is missing, respond with \"No evidence in provided images.\""
            "\n\nImage references:\n{references}"
        ).format(question=question, references="\n".join(image_captions))

        message_content = [{"type": "text", "text": user_prompt}] + image_contents

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": context_prompt},
                {"role": "user", "content": message_content},
            ],
            max_tokens=1000,
        )

        answer_text = response.choices[0].message.content.strip()
        if verbose:
            print("🧠 LLM Response:", answer_text)

        return answer_text

    except Exception as e:
        print(f"❌ Error processing images or getting response: {e}")
        return "Error occurred during processing."
