import datetime
import logging
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from definitions import CONFIG_PATH
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
from protollm.connectors import create_llm_connector

from ChemCoScientist.paper_analysis.chroma_db_operations import ChromaDBPaperStore
from ChemCoScientist.paper_analysis.prompts import sys_prompt_LLM
from ChemCoScientist.paper_analysis.question_processing import query_llm

load_dotenv(CONFIG_PATH)
from protollm.metrics import model_for_metrics

metrics_init_params = {
    "model": model_for_metrics,
    "verbose_mode": False,
    "async_mode": False,
}
correctness_metric = GEval(
    name="Correctness",
    evaluation_steps=[
        "If there are numeric values in the expected output, compare the numeric values from the actual output with the"
        " corresponding values from the expected output"
        " (if at least one value is missing or incorrect (except in cases where missing numeric values do not affect "
        " the accuracy of the information provided) -> LOW SCORE, else -> HIGH SCORE)",
        "Compare textual facts regardless of the formulations used and their order"
        " (if the facts are all OK -> LOW SCORE, else -> HIGH SCORE)",
        "Estimate the amount of filler text in the actual output"
        " (if there is a lot of it -> LOW SCORE, else -> HIGH SCORE)"
        "If the actual output does not contain an answer to the INPUT or reports that it cannot answer"
        " -> VERY LOW SCORE"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=model_for_metrics,
    async_mode=False
)
answer_relevancy = AnswerRelevancyMetric(**metrics_init_params)
faithfulness = FaithfulnessMetric(**metrics_init_params)
context_precision = ContextualPrecisionMetric(**metrics_init_params)
context_recall = ContextualRecallMetric(**metrics_init_params)
context_relevancy = ContextualRelevancyMetric(**metrics_init_params)

logging.basicConfig(level=logging.INFO)


def query_pure_llm(model_url: str, question: str) -> tuple:
    llm = create_llm_connector(model_url)

    messages = [
        SystemMessage(content=sys_prompt_LLM),
        HumanMessage(content=f"USER QUESTION: {question}")
    ]

    res = llm.invoke(messages)
    return res.content, res.response_metadata


def intersection_ratio(row, col1, col2):
    list1 = [elem.strip().lower() for elem in row[col1].split(";\n")]
    list2 = [elem.lower() for elem in eval(row[col2])]
    if len(list1) == 0:
        return 0
    intersection = set(list1) & set(list2)
    return len(intersection) / len(list1)


class Timer:
    def __init__(self):
        self.process_terminated = False

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    @property
    def start_time(self):
        return self.start

    @property
    def spent_time(self) -> datetime.timedelta:
        return datetime.datetime.now() - self.start_time

    @property
    def seconds_from_start(self) -> float:
        return round(self.spent_time.total_seconds(), 2)

    def __exit__(self, *args):
        return self.process_terminated


def pipeline_test_with_save(
        data: pd.DataFrame,
        metrics_to_calculate: list,
        m_name: str,
        m_url: str,
        version: float,
        out_dir: Path,
        paper_store: ChromaDBPaperStore
) -> pd.DataFrame:
    """Tests pipeline.

    Args:
        data: questions, correct context/answer etc.
        metrics_to_calculate: list of metrics to be calculated
        m_name: string with model name
        m_url: string with model URL and name
        version: test version
        out_dir: path to directory with results
        paper_store: connector to DB

    Returns: pandas DataFrame
    """
    print("Pipeline test is running...")
    out_dir.mkdir(parents=True, exist_ok=True)
    path_to_results = Path(out_dir, f"pipeline_test_{m_name}_v{version}.txt")
    path_to_df = Path(out_dir, f"pipeline_test_{m_name}_v{version}.csv")
    path_to_df_extended = Path(out_dir, f"pipeline_test_{m_name}_v{version}_extended.csv")

    columns = [
        "index", "question", "correct_paper", "correct_context", "papers_for_question", "txt_context_from_db",
        "img_context_from_db", "correct_answer", "answer_from_model", "context_retrieve_time",
        "answer_generation_time", "level", "category"
    ]
    for metric in metrics_to_calculate:
        columns.append(f"{metric.__name__}_score")
        columns.append(f"{metric.__name__}_reason")

    if path_to_df.exists():
        existing_df = pd.read_csv(path_to_df)
        clear_existing_df = existing_df.drop_duplicates(subset=["index"], keep=False)
        clear_existing_df.to_csv(path_to_df, index=False)
        processed_indices = clear_existing_df["index"].unique().tolist() if "index" in clear_existing_df.columns else []
        start_index = max(processed_indices) + 1 if processed_indices else 0
    else:
        existing_df = pd.DataFrame(columns=columns)
        existing_df.to_csv(path_to_df, index=False)
        start_index = 0

    for i, row in data.iterrows():
        if i < start_index:
            continue

        try:
            print(f"Processing question {i}")
            question = row["question"].replace('"', "'")
            correct_answer = row["correct_answer"]
            # correct_context = "\n".join(
            #     [str(row["correct_txt_context"]), str(row["correct_img_context"]), str(row["correct_table_context"])]
            # )
            correct_context = str(row["correct_txt_context"])

            row_data = {
                "index": i,
                "question": question,
                "correct_paper": row["file_name"],
                "correct_context": correct_context,
                "papers_for_question": None,
                "txt_context_from_db": None,
                "img_context_from_db": None,
                "correct_answer": correct_answer,
                "answer_from_model": "",
                "context_retrieve_time": None,
                "answer_generation_time": None,
                "level": row["level"],
                "category": row["category"]
            }

            for metric in metrics_to_calculate:
                row_data[f"{metric.__name__}_score"] = -1
                row_data[f"{metric.__name__}_reason"] = ""

            with Timer() as t:
                try:
                    txt_data, img_data, papers = paper_store.retrieve_context(
                        question
                    )  # for pure LLM test comment this method call
                    row_data["papers_for_question"] = papers['answer']  # for pure LLM test comment this line
                    row_data["context_retrieve_time"] = t.seconds_from_start
                except Exception as e:
                    print(f"Context retrieval failed: {str(e)}")
                    txt_context = ''
                ### for pure LLM test comment next block of code ###
                txt_context = ''
                img_paths = set()
                for idx, chunk in enumerate(txt_data, start=1):
                    txt_context += f"{idx}. Metadata: " \
                                   + str(chunk[2]) + "\nChunk: " \
                                   + chunk[1].replace("passage: ", "") + '\n\n'
                for chunk_meta in [chunk[2] for chunk in txt_data]:
                    img_paths.update(eval(chunk_meta["imgs_in_chunk"]))
                for img in img_data['metadatas'][0]:
                    img_paths.add(img['image_path'])
                ### -------------------------------------------- ###
                row_data["txt_context_from_db"] = txt_context
                row_data["img_context_from_db"] = img_paths

            with Timer() as t:
                try:
                    llm_res, _ = query_llm(m_url, question, txt_context, list(img_paths))
                    # comment out next line for pure LLM and comment previous line
                    # llm_res, _ = query_pure_llm(m_url, question)
                    row_data["answer_from_model"] = llm_res
                except Exception as e:
                    print(f"Answer generation failed: {str(e)}")
                    llm_res = ""
                row_data["answer_generation_time"] = t.seconds_from_start

            test_case = LLMTestCase(
                input=question,
                actual_output=llm_res,
                expected_output=correct_answer,
                context=[correct_context],
                retrieval_context=[txt_context],
            )
            for metric in metrics_to_calculate:
                try:
                    metric.measure(test_case)
                    row_data[f"{metric.__name__}_score"] = metric.score
                    row_data[f"{metric.__name__}_reason"] = metric.reason
                except Exception as e:
                    row_data[f"{metric.__name__}_score"] = -1
                    row_data[f"{metric.__name__}_reason"] = f"{type(e).__name__}: {str(e)}"

            row_df = pd.DataFrame([row_data])
            with open(path_to_df, 'a', newline='') as f:
                row_df.to_csv(f, header=f.tell() == 0, index=False)

        except Exception as e:
            print(f"Critical error processing question {i}: {str(e)}")
            if 'row_df' in locals():
                with open(path_to_df, 'a', newline='') as f:
                    row_df.to_csv(f, header=f.tell() == 0, index=False)
            raise

    result_df = pd.read_csv(path_to_df)
    result_df["total_time"] = (
            result_df["context_retrieve_time"] + result_df["answer_generation_time"]
    )
    result_df.loc[:, "correct_papers_or_not"] = result_df.apply(
        lambda row: intersection_ratio(row, "correct_paper", "papers_for_question"), axis=1
    )
    result_df.to_csv(path_to_df_extended, index=False)
    # Calculation of basic statistics for exec time and function selection
    average_correct_paper = result_df["correct_papers_or_not"].mean().round(2)
    avg_context_retrieve_time = result_df["context_retrieve_time"].mean().round(2)
    avg_ans_generation_time = result_df["answer_generation_time"].mean().round(2)
    avg_total_time = result_df["total_time"].mean().round(2)
    # Calculation of statistics for metrics
    metrics_score_columns = list(filter(lambda x: "score" in x, result_df.columns.tolist()))
    metrics_to_print = []
    for column in metrics_score_columns:
        result_df[column] = pd.to_numeric(result_df[column])
        avg_score = result_df[result_df[column] != -1][column].mean()
        failed_evaluations = result_df[result_df[column] == -1].shape[0]
        metrics_to_print.append(
            f"- Average {column} is {avg_score}. Number of unsuccessfully processed questions {failed_evaluations}"
        )
    short_metrics_result = "\n".join(metrics_to_print)

    to_print = f"""Average context retrieving time: {avg_context_retrieve_time}
Average answer generation time: {avg_ans_generation_time}
Average total time: {avg_total_time}
Average correct paper fraction: {average_correct_paper}
Short metrics results:
{short_metrics_result}"""

    with open(path_to_results, "w") as f:
        print(to_print, file=f)

    return result_df


if __name__ == "__main__":
    path_to_data = "../PaperAnalysis/questions/DataSet_FinalData.csv"
    out_dir = Path("../PaperAnalysis/test_results")
    all_questions = pd.read_csv(path_to_data)

    model_name = "gemini-2.0-flash-001"
    model_url = 'https://openrouter.ai/api/v1;google/gemini-2.0-flash-001'

    paper_store = ChromaDBPaperStore()  # for pure LLM test you don't need to use this instance

    v = 0.1
    pipeline_test_with_save(
        all_questions, [correctness_metric, context_recall], model_name, model_url, v, out_dir, paper_store
    )