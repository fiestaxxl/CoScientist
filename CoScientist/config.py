from dotenv import load_dotenv
import os
from pathlib import Path

from dataclasses import dataclass

ROOT_DIR = Path(__file__).parent.parent.absolute()  # go two levels up 
CONFIG_PATH = ROOT_DIR / "config.env"
load_dotenv(dotenv_path=CONFIG_PATH)


@dataclass
class LLMConfig:
    allowed_providers: list[str]
    service_key: str
    openai_api_key: str
    tavily_api_key: str

    main_url: str
    scenario_url: str
    main_model: str
    scenario_model: str

    service_url: str
    service_cc_url: str

    vision_url: str
    summary_url: str
    marker_model: str
    datasets_url: str
    deepeval_url: str


@dataclass
class StoragePaths:
    root_dir: str
    parse_results: str
    chroma_storage: str
    papers_storage: str
    ds_storage: str
    img_storage: str
    another_storage: str
    memory_db: str
    path_to_data: str
    path_to_cvae_checkpoint: str
    path_to_results: str
    path_to_temp_files: str
    my_papers: str

@dataclass
class HostsPorts:
    chroma_host: str
    embedding_host: str
    reranker_host: str
    opencChemie_host: str

    chroma_port: str
    embedding_port: str
    reranker_port: str
    opencChemie_port: str

@dataclass
class ToolsEndpoints:
    ml_tools_ip: str
    ml_tools_port: str
    dl_tools_ip: str
    dl_tools_port: str

@dataclass
class Collections:
    summaries: str
    texts: str
    images: str

@dataclass
class S3Config:
    use_s3: bool
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket_name: str

@dataclass
class OpikConfig:
    api_key: str
    url_override: str
    project_name: str

@dataclass
class Config:
    llm: LLMConfig
    storage: StoragePaths
    hosts_ports: HostsPorts
    tools: ToolsEndpoints
    collections: Collections
    s3: S3Config
    opik: OpikConfig

config = Config(
    llm=LLMConfig(
        allowed_providers=["google-vertex", "azure"],
        service_key=os.environ.get('LLM_SERVICE_KEY'),
        openai_api_key=os.environ.get('OPENAI_API_KEY'),
        tavily_api_key=os.environ.get('TAVILY_API_KEY'),
        main_url=os.environ.get('MAIN_LLM_URL'),
        scenario_url=os.environ.get('SCENARIO_LLM_URL'),
        main_model=os.environ.get('MAIN_LLM_MODEL'),
        scenario_model=os.environ.get('SCENARIO_LLM_MODEL'),
        service_url=os.environ.get('LLM_SERVICE_URL'),
        service_cc_url=os.environ.get('LLM_SERVICE_CC_URL'),
        vision_url=os.environ.get('VISION_LLM_URL'),
        summary_url=os.environ.get('SUMMARY_LLM_URL'),
        marker_model=os.environ.get('MARKER_LLM'),
        datasets_url=os.environ.get('DATASETS_LLM_URL'),
        deepeval_url=os.environ.get('DEEPEVAL_LLM_URL'),
    ),
    storage=StoragePaths(
        root_dir=ROOT_DIR,
        parse_results=os.environ.get('PARSE_RESULTS_PATH'),
        chroma_storage=os.environ.get('CHROMA_STORAGE_PATH'),
        papers_storage=os.environ.get('PAPERS_STORAGE_PATH'),
        ds_storage=os.environ.get('DS_STORAGE_PATH'),
        img_storage=os.environ.get('IMG_STORAGE_PATH'),
        another_storage=os.environ.get('ANOTHER_STORAGE_PATH'),
        memory_db=os.environ.get('MEMORY_DB_PATH'),
        path_to_data=os.environ.get('PATH_TO_DATA'),
        path_to_cvae_checkpoint=os.environ.get('PATH_TO_CVAE_CHECKPOINT'),
        path_to_results=os.environ.get('PATH_TO_RESULTS'),
        path_to_temp_files=os.environ.get('PATH_TO_TEMP_FILES'),
        my_papers=os.environ.get('MY_PAPERS_PATH'),
    ),
    hosts_ports=HostsPorts(
        chroma_host=os.environ.get('CHROMA_HOST'),
        embedding_host=os.environ.get('EMBEDDING_HOST'),
        reranker_host=os.environ.get('RERANKER_HOST'),
        opencChemie_host=os.environ.get('OPENCHEMIE_HOST'),
        chroma_port=os.environ.get('CHROMA_PORT'),
        embedding_port=os.environ.get('EMBEDDING_PORT'),
        reranker_port=os.environ.get('RERANKER_PORT'),
        opencChemie_port=os.environ.get('OPENCHEMIE_PORT'),
    ),
    tools=ToolsEndpoints(
        ml_tools_ip=os.environ.get('ML_TOOLS_IP'),
        ml_tools_port=os.environ.get('ML_TOOLS_PORT'),
        dl_tools_ip=os.environ.get('DL_TOOLS_IP'),
        dl_tools_port=os.environ.get('DL_TOOLS_PORT'),
    ),
    collections=Collections(
        summaries=os.environ.get('SUMMARIES_COLLECTION_NAME'),
        texts=os.environ.get('TEXTS_COLLECTION_NAME'),
        images=os.environ.get('IMAGES_COLLECTION_NAME'),
    ),
    s3=S3Config(
        use_s3=os.environ.get('USE_S3') == 'True',
        endpoint_url=os.environ.get('ENDPOINT_URL'),
        access_key=os.environ.get('ACCESS_KEY'),
        secret_key=os.environ.get('SECRET_KEY'),
        bucket_name=os.environ.get('LLMBUCKET_NAME_SERVICE_KEY'),
    ),
    opik=OpikConfig(
        api_key=os.environ.get('OPIK_API_KEY'),
        url_override=os.environ.get('OPIK_URL_OVERRIDE'),
        project_name=os.environ.get('OPIK_PROJECT_NAME'),
    )
)

