import json
import os
import subprocess
import sys
import time
from multiprocessing import Process
from typing import List, Tuple, Union

import pandas as pd
import requests
from smolagents import tool

from ChemCoScientist.tools.utils import filter_valid_strings

# TODO: get from load_env
#conf = {"url_pred": "http://10.32.2.2:293", "url_gen": "http://10.32.2.2:293"}
conf = {"url_pred": f'{os.environ.get("ML_TOOLS_IP")}:{os.environ.get("ML_TOOLS_PORT")}', "url_gen": f'{os.environ.get("ML_TOOLS_IP")}:{os.environ.get("ML_TOOLS_PORT")}'}


@tool
def get_state_from_server(url: str = "pred") -> Union[dict, str]:
    """
    Retrieves the current status and information about available models from the server.
    
    This method checks the status of models – whether they are in a training state, have been trained, or are pending – and provides details about their descriptions and associated metrics. It facilitates monitoring the readiness and state of different model deployments. In case of a server error, it returns an appropriate error message.
    
    Args:
        url (str, optional): Specifies the server endpoint. Use 'pred' for predictive models or 'gen' for generative models. Defaults to "pred".
    
    Returns:
        dict or str: A dictionary containing model state information (status, description, metrics) if successful.  Returns a string "Server error" if a server-side error (status code 500) occurs.
    """
    if url == "pred":
        url = conf["url_pred"]
    else:
        url = conf["url_gen"]

    url_ = url.split("http://")[1]
    resp = requests.get("http://" + url_.split("/")[0] + "/check_state")
    if resp.status_code == 500:
        print(f"Server error:{resp.status_code}")
        return "Server error"
    state = json.loads(resp.content)
    return state["state"]


@tool
def get_case_state_from_server(case: str, url: str = "pred") -> Union[dict, str]:
    """
    Retrieve the current status and metrics for a specific case/model from the server.
    
    This method queries the server to determine the state of a given case (e.g., 'Training', 'Trained', or details of an error if one occurred). It allows checking the progress or result of a model training or prediction process.
    
    Args:
        case (str): The name of the case/model to query.
        url (str, optional):  Specifies the server endpoint. Use 'pred' for predictive models and 'gen' for generative models. Defaults to "pred".
    
    Returns:
        dict: A dictionary containing the case's state and metrics if found.
        str: An error message if the case is not found or if the server returns an error.  If the 'status' key in the dictionary contains text instead of 'Training', 'Trained', or None, it indicates an error condition.
    """
    if url == "pred":
        url = conf["url_pred"]
    else:
        url = conf["url_gen"]

    url_ = url.split("http://")[1]
    resp = requests.get("http://" + url_.split("/")[0] + "/check_state")
    if resp.status_code == 500:
        print(f"Server error:{resp.status_code}")
        return "Server error"
    state = json.loads(resp.content)
    try:
        return state["state"][case]
    except:
        return f"Case with name: {case} not found"


@tool
def predict_prop_by_smiles(
    smiles_list: List[str], case: str = "no_name_case", timeout: int = 20
) -> Tuple[requests.Response, dict]:
    """
    Predicts molecular properties using pre-trained machine learning models. This function facilitates rapid assessment of molecule characteristics by sending SMILES representations to a prediction server.
    
    Args:
        smiles_list (List[str]): A list of molecules represented in SMILES format for property prediction.
        case (str, optional): Specifies the model to use for prediction. Defaults to "no_name_case".  Available models can be retrieved using 'get_state_from_server'.
        timeout (int, optional): Sets a time limit (in minutes) for the prediction request. Defaults to 20.
    
    Returns:
        Tuple[requests.Response, dict]: A tuple containing the HTTP response from the prediction server and the parsed JSON data representing the predicted properties.
    """
    url = conf["url_pred"] + "/predict_ml"
    params = {"case": case, "smiles_list": smiles_list, "timeout": timeout}
    resp = requests.post(url, json.dumps(params))
    return resp, resp.json()


def train_gen_with_data(
    case="no_name",
    data_path="./data_dir_for_coder/kras_g12c_affinity_data.xlsx",  # path to client data folder
    feature_column=["smiles"],
    target_column=[
        "docking_score",
        "QED",
        "Synthetic Accessibility",
        "PAINS",
        "SureChEMBL",
        "Glaxo",
        "Brenk",
        "IC50",
    ],  # All propreties from dataframe you want to calculate in the end
    regression_props=[
        "docking_score"
    ],  # Column name with data for regression tasks (That not include in calculcateble propreties)
    classification_props=[],  # Column name with data for classification tasks (That not include in calculcateble propreties)
    description="Descrption not provided",
    timeout=5,  # min
    url: str = conf["url_gen"] + "/train_gen_models",
    fine_tune: bool = True,
    n_samples=10,
    **kwargs,
):
    """
    Trains a generative deep learning model to learn relationships between molecular structures and their properties from a provided dataset.
    
    This method prepares data from a CSV or Excel file, containing SMILES strings and associated properties, and sends it to a server for model training.
    The trained model can then be used to generate new molecules with desired characteristics.
    
    Args:
        case (str): A name for the training case, used for identification.
        data_path (str): Path to the data file (CSV or Excel) containing molecular data. The file must include SMILES strings.
        feature_column (list): List of column names representing the molecular features (inputs). Defaults to ['smiles'].
        target_column (list): List of column names representing the properties to be predicted (targets).
        regression_props (list): List of column names representing properties to be predicted using regression.
        classification_props (list): List of column names representing properties to be predicted using classification.
        description (str): A description of the model or training case.
        timeout (int): Training timeout in minutes.
        url (str): URL of the server endpoint for training the model. Defaults to a configured URL.
        fine_tune (bool):  A flag indicating whether to fine-tune an existing model. Defaults to True.
        n_samples (int): Number of samples to use for validation during training. Defaults to 10.
        **kwargs: Additional keyword arguments to be passed to the training process.
    
    Returns:
        None
    """
    start_time = time.time()
    try:
        df = pd.read_csv(
            data_path
        ).to_dict()  # Transfer df to dict for server data transfer
    except:
        df = pd.read_excel(data_path).to_dict()

    params = {
        "case": case,
        "data": df,
        "target_column": target_column,
        "feature_column": feature_column,
        "timeout": timeout,
        "description": description,
        "regression_props": regression_props,
        "classification_props": classification_props,
        "fine_tune": fine_tune,
        "n_samples": n_samples,
        **kwargs,
    }

    p = Process(target=requests.post, args=[url, json.dumps(params)])
    p.start()

    time.sleep(4)
    print("--- %s seconds ---" % (time.time() - start_time))


def train_ml_with_data(
    case="No_name",
    data_path="automl/data/data_4j1r.csv",  # path to client data folder
    feature_column=["Smiles"],
    target_column=[
        "Docking score"
    ],  # All propreties from dataframe you want to calculate in the end,
    regression_props=["Docking score"],
    classification_props=[],
    description="",
    timeout=5,
) -> Union[bool, str]:
    """
    Trains a machine learning model using a provided dataset to enable chemical property prediction.
    
    This function prepares data from a CSV or Excel file and sends it to a remote server 
    for model training. The process runs asynchronously to avoid blocking the main application. 
    The trained model can then be used to predict properties of new chemical compounds.
    
    Args:
        case (str, optional): Name of the model being trained. Defaults to "No_name".
        data_path (str, optional): Path to the CSV or Excel file containing the dataset. 
                                    Defaults to "automl/data/data_4j1r.csv".
        feature_column (list, optional): The name of the column containing the input features (e.g., SMILES strings). 
                                          Defaults to ["Smiles"].
        target_column (list, optional):  List of column names representing the properties to be predicted. 
                                          Defaults to ["Docking score"].
        regression_props (list, optional): Column names containing data for regression tasks.  
                                            Skip if no regression is needed. Defaults to ["Docking score"].
        classification_props (list, optional): Column names containing data for classification tasks.
                                              Skip if no classification is needed. Defaults to [].
        timeout (int, optional): The timeout duration (in minutes) for the request. Defaults to 5.
        description (str): A description of the model or training case. Defaults to "".
    
    Returns:
        bool: True if the training process was initiated successfully, False otherwise.
    """
    start_time = time.time()
    try:
        df = pd.read_csv(
            data_path
        ).to_dict()  # Transfer df to dict for server data transfer
    except:
        df = pd.read_excel(data_path).to_dict()
    start_time = time.time()
    params = {
        "case": case,
        "data": df,
        "target_column": target_column,
        "feature_column": feature_column,
        "timeout": timeout,
        "description": description,
        "regression_props": regression_props,
        "classification_props": classification_props,
    }

    p = Process(
        target=requests.post,
        args=[f"{conf['url_pred']}/train_ml", json.dumps(params)],
    )
    p.start()

    time.sleep(10)
    p.terminate()

    print("--- %s seconds ---" % (time.time() - start_time))

    return True


def ml_dl_training(
    case: str,
    path: str,
    feature_column=["canonical_smiles"],
    target_column=["docking_score"],
    regression_props=["docking_score"],
    classification_props=[],
):
    """
    Trains machine learning and deep learning models for a given case.
    
    This method initiates the training of machine learning (ML) and generative (Gen) models
    using provided data, and actively monitors their progress by polling a server for completion status.
    This ensures models are fully trained before proceeding, enabling tasks such as property prediction
    and molecule generation based on the data.
    
    Args:
        case (str): The identifier for the case to train the models for.
        path (str): The path to the data used for training.
        feature_column (list[str], optional): The name(s) of the column(s) to use as features. Defaults to ["canonical_smiles"].
        target_column (list[str], optional): The name(s) of the column(s) to use as targets. Defaults to ["docking_score"].
        regression_props (list[str], optional): The name(s) of the regression properties. Defaults to ["docking_score"].
        classification_props (list[str], optional): The name(s) of the classification properties. Defaults to [].
    
    Returns:
        None
    """
    ml_ready = False
    train_ml_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
    )
    print("Start training ml model for case: ", case)
    while not ml_ready:
        print("Training ml-model in progress for case: ", case)
        st = get_case_state_from_server(case, "pred")
        if isinstance(st, dict):
            if st["ml_models"]["status"] == "Trained":
                ml_ready = True
        time.sleep(60)

    train_gen_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
        # TODO: rm after testing automl pipeline
        # epoch=1,
    )
    print("Start training gen model for case: ", case)


def ml_dl_training(
    case: str,
    path: str,
    feature_column=["canonical_smiles"],
    target_column=["docking_score"],
    regression_props=["docking_score"],
    classification_props=[],
):
    """
    Trains machine learning and deep learning models for a given case.
    
    This method trains both machine learning and generative models on provided data.
    It first trains a machine learning model, monitors its progress until completion,
    and then proceeds to train a generative model using the same data.
    This sequential training allows leveraging the initial model to inform and improve
    the generative process for downstream tasks.
    
    Args:
        case (str): The identifier for the training case.
        path (str): The path to the data used for training.
        feature_column (list, optional): The name of the column containing features. Defaults to ["canonical_smiles"].
        target_column (list, optional): The name of the column containing target variables. Defaults to ["docking_score"].
        regression_props (list, optional): List of properties to use for regression. Defaults to ["docking_score"].
        classification_props (list, optional): List of properties to use for classification. Defaults to [].
    
    Returns:
        None
    """
    ml_ready = False
    train_ml_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
    )
    print("Start training ml model for case: ", case)
    while not ml_ready:
        print("Training ml-model in progress for case: ", case)
        st = get_case_state_from_server(case, "pred")
        if isinstance(st, dict):
            if st["ml_models"]["status"] == "Trained":
                ml_ready = True
        time.sleep(60)

    train_gen_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
        # TODO: rm after testing automl pipeline
        # epoch=1,
    )
    print("Start training gen model for case: ", case)


def ml_dl_training(
    case: str,
    path: str,
    feature_column=["canonical_smiles"],
    target_column=["docking_score"],
    regression_props=["docking_score"],
    classification_props=[],
):
    """
    Trains machine learning and deep learning models for a given case.
    
    This method trains both machine learning and generative models based on the provided data, 
    with the generative model building upon the insights gained from the initial machine learning phase.
    It continuously checks the status of the ML model training and then proceeds to train the generative model once the ML model is ready.
    
    Args:
        case (str): The identifier for the case to train models for.
        path (str): The path to the data used for training.
        feature_column (list, optional): The column(s) to use as features for the models. Defaults to ["canonical_smiles"].
        target_column (list, optional): The column(s) to use as targets for the models. Defaults to ["docking_score"].
        regression_props (list, optional): The properties to use for regression modeling. Defaults to ["docking_score"].
        classification_props (list, optional): The properties to use for classification modeling. Defaults to [].
    
    Returns:
        None
    """
    ml_ready = False
    train_ml_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
    )
    print("Start training ml model for case: ", case)
    while not ml_ready:
        print("Training ml-model in progress for case: ", case)
        st = get_case_state_from_server(case, "pred")
        if isinstance(st, dict):
            if st["ml_models"]["status"] == "Trained":
                ml_ready = True
        time.sleep(60)

    train_gen_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
        # TODO: rm after testing automl pipeline
        # epoch=1,
    )
    print("Start training gen model for case: ", case)


@tool
def just_ml_training(
    case: str,
    path: str,
    feature_column: list = ["canonical_smiles"],
    target_column: list = ["docking_score"],
    regression_props: list = ["docking_score"],
    classification_props: list = [],
) -> bool:
    """
    Trains a machine learning model for predicting molecular properties.
    
    This method prepares data, validates its integrity, and initiates the training process using the provided dataset and specified properties.
    
    Args:
        case (str): A unique identifier for the model being trained.
        path (str): The file path to the dataset (CSV or Excel).
        feature_column (list, optional):  The name(s) of the column(s) containing the molecular features. Defaults to ["canonical_smiles"].
        target_column (list): The name(s) of the column(s) representing the properties to be predicted.  Must contain at least one element.
        regression_props (list, optional): The name(s) of the column(s) used for regression tasks. Defaults to ["docking_score"].  Should duplicate `feature_column` if used.
        classification_props (list, optional): The name(s) of the column(s) used for classification tasks. Defaults to [].  Set to an empty list if classification is not required.
    
    Returns:
        bool: True upon successful initiation of the training process.  Raises exceptions if data validation fails.
    """

    if regression_props == [] and classification_props == []:
        regression_props = target_column
    if len(target_column) < 1:
        raise ValueError(
            "target_column is empty! You must set value. For example = ['IC50']"
        )
    if len(feature_column) < 1:
        raise ValueError(
            "feature_column is empty! You must set value. For example = ['smiles']"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_excel(path)

    if len(df.values.tolist()) < 300:
        raise ValueError(
            "Training on this data is impossible. The dataset is too small!"
        )

    for column in feature_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )
    for column in target_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )

    # delete molecules eith len more 200
    clear_df = filter_valid_strings(df, feature_column[0])
    if path.split(".")[-1] == "csv":
        clear_df.to_csv(path)
    else:
        clear_df.to_excel(path)

    train_ml_with_data(
        case=case,
        data_path=path,
        feature_column=feature_column,
        target_column=target_column,
        regression_props=regression_props,
        classification_props=classification_props,
    )
    print("Start training ml model for case: ", case)
    return True


@tool
def generate_mol_by_case(
    case: str = "Alzheimer",
    n_samples: int = 10,
) -> dict:
    """
    Generates molecules based on a specified model and quantity.
    
    This method facilitates the creation of new molecular structures using pre-trained generative models. It sends a request to a remote server to perform the generation and returns the results.
    
    Args:
        case (str, optional): The name of the generative model to use.  Available model names can be retrieved using `get_state_from_server`. Defaults to "Alzheimer".
        n_samples (int, optional): The number of molecules to generate. Defaults to 10.
    
    Returns:
        dict: A dictionary containing the generated molecules.
    """
    url = conf["url_gen"] + "/generate_gen_models_by_case"

    params = {
        "case": case,
        "n_samples": n_samples,
    }
    start_time = time.time()
    resp = requests.post(url, data=json.dumps(params))
    print("--- %s seconds ---" % (time.time() - start_time))
    try:
        return json.loads(resp.json())
    except:
        resp.json()

def generate_with_gan(num=100):
    params = {'case_': "Alzheimer",'numb_mol': num}
    url = conf["url_gen"] + '/gan_case_generator'
    resp = requests.post(url, json.dumps(params))
    return resp, resp.json()

@tool
def generate_with_gan_cyclic(
    num: int = 10,
    properties_conditions: dict[str, str] | None = None,
    num_tries: int = 5,
    success_rate: float = 0.5,
    maximum_error: float = 0.1,
    batch_size: int = 10,
) -> list[str]:
    """
    Generate molecules with a GAN in iterative batches and filter them by property conditions. Use this a main tool for molecule generation.

    This tool repeatedly requests batches of molecules from a GAN-backed generator and
    selects those that satisfy user-defined property constraints. It stops early once
    at least `num` molecules pass the filters or reaches the global budget of
    `num * num_tries` generated candidates. The function returns a list of SMILES and
    prints a short English summary with basic statistics (matches on the first batch,
    total matches after retries, unsupported properties, etc.).

    Format of `properties_conditions` (string constraints):
      - Each property maps to a string: "<op><value>", where <op> is one of {">=", "<=", "==", ">", "<"}
        and <value> is a float. Examples:
          {"QED": ">=0.7", "Synthetic Accessibility": "<=3.0", "PAINS": "==0"}
      - Equality uses a relative tolerance ±`maximum_error`. For example, "==4" with
        maximum_error=0.05 accepts values in [3.8, 4.2].
      - For inequalities (>=, <=, >, <), no tolerance is applied.

    Guidance for agent message to conditions mapping (examples):
      - "high QED"        → {"QED": ">=0.7"} (use 0.7–0.8 as a common threshold)
      - "very high QED"   → {"QED": ">=0.8"}
      - "easy to synthesize" → {"Synthetic Accessibility": "<=3.0"}
      - "no PAINS alerts" → {"PAINS": "==0"}
      - "strong docking"  → {"docking_score": "<=-7.0"}  (more negative is better)
      - "low IC50"        → {"IC50": "<=1.0"}  (adjust to the project’s scale/units)

    Args:
        num (int, optional): Target number of molecules to return (after filtering).
        properties_conditions (dict[str, str], optional): Mapping from property name to
            a condition string ("<op><value>"), e.g. {"QED": ">=0.7"}.
        num_tries (int, optional): Global generation multiplier. At most `num * num_tries`
            candidates are generated across all batches.
        success_rate (float, optional): Expected success rate to include in the printed summary
            (informational only; not enforced).
        maximum_error (float, optional): Relative tolerance applied only to '==' comparisons.
        batch_size (int, optional): Batch size for each GAN generation request.

    Returns:
        list[str]: A list of SMILES strings. If no molecule satisfies the constraints,
                   returns up to `num` best-effort molecules from all generated batches.

    Behavior:
        - Filters only by properties present in the predictor output; unknown properties
          are ignored and reported in the printed summary.
        - Stops early once `num` passing molecules are found.

    Raises:
        ValueError: If a condition string has an unsupported format (must be one of
                    '>=x', '<=x', '==x', '>x', '<x').

    Examples:
        # Agent prompt → tool call
        "Generate 10 molecules with high QED and no PAINS alerts"
          → num=10,
            properties_conditions={"QED": ">=0.7", "PAINS": "==0"}

        # Strong docking and easy synthesis
        "Generate 20 easy-to-synthesize molecules with strong docking"
          → num=20,
            properties_conditions={"Synthetic Accessibility": "<=3.0",
                                   "docking_score": "<=-7.0"}
    """
    def parse_condition(cond_str):
        s = str(cond_str).strip()
        for op in ("<=", ">=", "==", "<", ">"):
            if op in s:
                return op, float(s.split(op)[1].strip())
        raise ValueError(f"Unsupported condition format: {cond_str}")

    def make_check(op, thr):
        if op == "==":
            low, high = thr * (1 - maximum_error), thr * (1 + maximum_error)
            return lambda v: v is not None and low <= v <= high
        if op == ">=":
            return lambda v: (v is not None) and (v >= thr)
        if op == "<=":
            return lambda v: (v is not None) and (v <= thr)
        if op == ">":
            return lambda v: (v is not None) and (v > thr)
        if op == "<":
            return lambda v: (v is not None) and (v < thr)
        raise ValueError(f"Unsupported operator: {op}")

    max_total = int(num) * num_tries
    all_samples, selected = [], []

    first_batch = generate_with_gan(num=batch_size)[1]
    first_batch = json.loads(first_batch)
    props_first = first_batch
    first_batch = first_batch['Smiles']
    all_samples.extend(first_batch)

    available = set(props_first.keys())
    unsupported = [p for p in properties_conditions if p not in available]
    supported = {p: properties_conditions[p] for p in properties_conditions if p in available}

    checks = {p: make_check(*parse_condition(c)) for p, c in supported.items()}
    def passes(idx, props_dict):
        for p, check in checks.items():
            vals = props_dict.get(p)
            if vals is None or idx >= len(vals) or not check(vals[idx]):
                return False
        return True

    first_ok = []
    for i, smi in enumerate(first_batch):
        if passes(i, props_first):
            first_ok.append(smi)
    selected.extend(first_ok)

    if len(selected) >= num:
        result = selected[:num]
        summary = (
            f"Initial pass: {len(first_ok)} matched of {len(first_batch)}. "
            f"After retries: {len(selected)} matched of {len(all_samples)}. "
            f"Target={num}, generated={len(all_samples)}, "
            f"success_rate_observed={len(selected)/len(all_samples):.2f}, "
            f"expected_success_rate={success_rate:.2f}. "
            + (f"Unsupported properties: {unsupported}. " if unsupported else "")
            + (f"Equality tolerance ±{int(maximum_error*100)}%." if '==' in ''.join(properties_conditions.values()) else "")
        )
        return result, summary

    while len(all_samples) < max_total and len(selected) < num:
        to_gen = min(num, max_total - len(all_samples))
        if to_gen <= 0:
            break
        batch = generate_with_gan(num=batch_size)[1]
        batch = json.loads(batch)
        props_batch = batch
        batch = batch['Smiles']
        all_samples.extend(batch)

        for i, smi in enumerate(batch):
            if passes(i, props_batch):
                selected.append(smi)
                if len(selected) >= num:
                    break

    if len(selected) >= num:
        result = selected[:num]
    else:
        result = all_samples[:min(num, len(all_samples))]

    summary = (
        f"Initial pass: {len(first_ok)} matched of {len(first_batch)}. "
        f"After retries: {len(selected)} matched of {num}. "
        f"Target={num}, generated={len(all_samples)}, "
        f"success_rate_observed={(len(selected)/num if all_samples else 0):.2f}, "
        f"expected_success_rate={success_rate:.2f}. "
        + (f"Unsupported properties: {unsupported}. " if unsupported else "")
        + (f"Equality tolerance ±{int(maximum_error*100)}%." if '==' in ''.join(properties_conditions.values()) else "")
    )
    print(summary)
    return result

@tool
def run_ml_dl_training_by_daemon(
    case: str,
    path: str,
    feature_column: list = ["smiles"],
    target_column: list[str] = ["docking_score"],
    regression_props: list[str] = ["docking_score"],
    classification_props: list = [],
) -> Union[bool, str]:
    """
    Initiates a machine learning and deep learning training process in the background, utilizing the provided dataset and configuration. 
    This process prepares models for predicting chemical properties or classifying molecules based on input features. 
    The training status can be monitored separately using the "get_state_case_from_server" function.
    
    Args:
        case (str): A unique identifier for the training case, used to track the process.
        path (str): The file path to the dataset, which should be a CSV or Excel file.
        feature_column (list, optional):  A list of column names representing the input features for the model. Defaults to ["smiles"].
        target_column (list, optional): A list of column names specifying the properties to be predicted or classified. This list must not be empty. Defaults to ["docking_score"].
        regression_props (list, optional): A list of column names used for regression tasks. This list should generally align with the feature columns. Defaults to ["docking_score"].
        classification_props (list, optional): A list of column names used for classification tasks. Use an empty list if classification is not required. Defaults to [].
    
    Returns:
        bool: True if the training process was successfully initiated, False otherwise.
        str:  If the process fails, returns a string describing the error.
    
    Raises:
        ValueError: If target_column or feature_column are empty, or if specified columns are not found in the dataset, or if the dataset is too small (less than 300 samples).
        FileNotFoundError: If the specified data file does not exist.
    
    Note: At least one of regression_props or classification_props must be specified for the training to proceed.
    """
    if isinstance(feature_column, str):
        feature_column = [feature_column]
    if isinstance(target_column, str):
        target_column = [target_column]

    if regression_props == [] and classification_props == []:
        regression_props = target_column
    if len(target_column) < 1:
        raise ValueError(
            "target_column is empty! You must set value. For example = ['IC50']"
        )
    if len(feature_column) < 1:
        raise ValueError(
            "feature_column is empty! You must set value. For example = ['smiles']"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_excel(path)

    if len(df.values.tolist()) < 300:
        raise ValueError(
            "Training on this data is impossible. The dataset is too small!"
        )

    for column in feature_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )
    for column in target_column:
        if column not in df.columns.tolist():
            raise ValueError(
                f'No "{column}" column in data! Change argument and run again! Avilable: '
                + str(df.columns.tolist())
            )

    # delete molecules eith len more 200
    clear_df = filter_valid_strings(df, feature_column[0])
    if path.split(".")[-1] == "csv":
        clear_df.to_csv(path)
    else:
        clear_df.to_excel(path)

    cmd = [
        sys.executable,
        "-c",
        (
            "from ChemCoScientist.tools.ml_tools import ml_dl_training;"
            "ml_dl_training("
            f"case='{case}',"
            f"path='{path}',"
            f"feature_column={feature_column},"
            f"target_column={target_column},"
            f"regression_props={regression_props},"
            f"classification_props={classification_props}"
            ")"
        ),
    ]

    try:
        cwd_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # get root dir
        
        subprocess.Popen(
            cmd,
            stdout=open("/tmp/ml_training.log", "a"),
            stderr=open("/tmp/ml_training.err", "a"),
            cwd=cwd_path,
        )
        time.sleep(5)
        return True
    except Exception as e:
        print(f"Failed to start process: {e}", file=sys.stderr)
        return False


agents_tools = [
    run_ml_dl_training_by_daemon,
    get_case_state_from_server,
    get_state_from_server,
    generate_mol_gan,
    # generate_mol_by_case,
    predict_prop_by_smiles,
]
if __name__ == "__main__":
    run_ml_dl_training_by_daemon('sars_cov', './data_store/datasets/users_dataset.csv', 'smiles', 'IC50', ['IC50'])
