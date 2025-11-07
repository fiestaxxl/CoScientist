import os, json, base64
import pandas as pd
from dotenv import load_dotenv
import fitz
from io import BytesIO, StringIO
from PIL import Image
from langchain_core.messages import SystemMessage, HumanMessage
from protollm.connectors import create_llm_connector

from ChemCoScientist.chemical_utils.openchemie_functions import extract_molecules_from_figure
from ChemCoScientist.paper_analysis.settings import allowed_providers
from ChemCoScientist.paper_analysis.prompts import extract_mol_properties_prompt

from definitions import CONFIG_PATH

load_dotenv(CONFIG_PATH)

VISION_LLM_URL = os.environ["VISION_LLM_URL"]

def convert_pdf_pages_to_images(path: str) -> list:
    """Converts PDF pages into fitz.Pixmap images."""
    doc = fitz.open(path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=500)
        images.append(pix)
    return images
        
def extract_smiles_from_images(images: list) -> list:
    """Uses OpenChemIE tool to extract molecular SMILES and ID from PDF screenshots."""
    results = []
    for img in images:
        pil_image = Image.open(BytesIO(img.tobytes("png")))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        image_file = buffered.getvalue()
        res = extract_molecules_from_figure(image=image_file)
        results.append(res)
    return results

def mols_to_csv(results):
    """"Formats OpenChemIE JSONs with molecular SMILES and IDs into pandas DataFrame."""
    df = pd.DataFrame()
    
    all_refs = []
    all_smiles = []
    for res in results:
        res = res[0]
        corefs = res["corefs"]
        mols_idxs = [i[0] for i in corefs]
        refs_idxs = [i[1] for i in corefs]
        smiles = [res["bboxes"][i]["smiles"] for i in mols_idxs]
        refs = [res["bboxes"][i]["text"] for i in refs_idxs]
        for i in range(len(corefs)):
            ref = ";".join(refs[i]) if refs[i] != [] else "Unknown ID"
            all_refs.append(ref)
            all_smiles.append(smiles[i])
    df["id"] = all_refs
    df["smiles"] = all_smiles

    return df

def extract_props(model_url: str, question: str, pdfs: list) -> dict:
    """
    Queries a language model with a question and a list of PDF documents to collect a dataset of
    molecules and their properties from scientific papers.
    """
    from ChemCoScientist.frontend.utils import update_activity

    if pdfs:
        update_activity(os.path.dirname(pdfs[0]))
    llm = create_llm_connector(model_url)

    content = []

    for paper_pdf in pdfs:
        with open(paper_pdf, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        paper_part = {
            "type": "file",
            "file": {
                "filename": paper_pdf,
                "file_data": f"data:application/pdf;base64,{base64_pdf}",
            },
        }
        content.append(paper_part)

    text_part = {"type": "text", "text": f"USER QUESTION: {question}"}
    content.append(text_part)
    from langchain_core.messages import HumanMessage

    messages = [
        SystemMessage(content=extract_mol_properties_prompt),
        HumanMessage(content=content)
    ]

    res = llm.invoke(messages)
    clean_text = res.content.split("```csv")[-1].split("```")[0].strip()
    df = pd.read_csv(StringIO(clean_text))
    return df

def extract_mols_prop_dataset(model_url: str, question: str, pdfs: list) -> pd.DataFrame:
    """
    Extracts a dataset with molecular SMILES and properties from PDF documents
    by calling OpenChemIE tool and quering a language model.
    
    Args:
        model_url (str): The URL of the language model to use for querying.
        question (str): The question to ask the language model.
        pdfs (list): A list of paths to PDF documents.

    Returns:
        pd.DataFrame: A dataset with molecular IDs, SMILES and required properties extracted from provided PDF documents.
    """
    all_datasets = []
    for pdf in pdfs:
        images = convert_pdf_pages_to_images(pdf)
        results = extract_smiles_from_images(images)
        mols_df = mols_to_csv(results)
        mols_df['id'] = mols_df['id'].astype(str)
        props_df = extract_props(model_url, question, [pdf])
        props_df['id'] = props_df['id'].astype(str)
        merged_df = pd.merge(props_df, mols_df, on="id", how="inner")
        merged_df["source"] = os.path.basename(pdf)
        all_datasets.append(merged_df)
        
    combined_dataset = pd.concat(all_datasets, ignore_index=True)
    
    return combined_dataset


# if __name__ == "__main__":
#     pdfs = [r"C:\Users\computer\Documents\GitHub\CoScientist\ChemCoScientist\paper_analysis\papers\187152108785908820.pdf"]
#     question = "Collect a dataset of molecules and their MIC values against Staphylococcus aureus."
#     extract_mols_prop_dataset(VISION_LLM_URL, question, pdfs)