import pandas as pd  # noqa: I001
import argparse , time ,re
import numpy as np
from sklearn.ensemble import IsolationForest
from canonical_column_names import CANONICAL_FIELDS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import datetime as dt
from jinja2 import Environment ,FileSystemLoader
import webbrowser




##### data #####

#Messy_Employee_dataset.csv

def main():
    ##### timer start #####
    run_date = dt.datetime.now().astimezone()
    start_ = time.perf_counter()


    ##### argparser #####

    parser = argparse.ArgumentParser()
    parser.add_argument("--file",help="Path for the target dirty csv")
    # parser.add_argument("--output")
    # parser.add_argument("--output", default="clean/cleaned.csv",
    #                     help="Path for the cleaned CSV file ,default :clean/cleaned.csv")
    # parser.add_argument("--review", default="review/review_these_rows.csv",
    #                     help="Path for the manual-review CSV , default :clean/cleaned.csv")
    parser.add_argument("--threshold", default=0.75,
                            help="default :0.75")
    parser.add_argument("--contamination", default=0.05, type=float,
                    help="IsolationForest contamination rate, (0.0, 0.5]. Default: 0.05")
    args = parser.parse_args()
    input_file = args.file


    ##### load csv file #####
    def load_csv_file(file_name):
        try :
            dirty_df = pd.read_csv(f"dirty_data/{file_name}",header=None)

        except FileNotFoundError :
            print(f"file not found : dirty_data/{file_name}")

        except pd.errors.EmptyDataError :
            print(f"file is empty : dirty_data/{file_name}")

        except pd.errors.ParserError :
            print("Wrong delimiter (e.g., commas vs. semicolons),mismatched quotes, or a row has more columns than the header.")

        else :

            dirty_df.columns = dirty_df.iloc[0]
            dirty_df = dirty_df.iloc[1:]

            # print(f"-number of rows and columns :{dirty_df.shape} \n-first 5 rows :\n{dirty_df.head()} \ncolumns types :\n{dirty_df.dtypes}\n{dirty_df.describe()}")

            return dirty_df , dirty_df.shape
    dirty_df ,dirty_df_shape = load_csv_file(args.file)


    ##### cleaning clolumn names #####
    def clean_column_names(df) :
        # print(f"befor the cleaning {df.columns}")
        clean_columns = []

        for ind , col in  enumerate(df.columns):
            cleaned = col.strip().lower()
            cleaned_v2 = re.sub(r"[ -]", "_",cleaned)
            cleaned_v3 = re.sub(r"[^a-zA-Z0-9_]", "",cleaned_v2)
            cleaned_v4 = re.sub(r"_+" , "_" , cleaned_v3)
            if not cleaned_v4 :
                cleaned_v4 = f"column_{ind}"

            clean_columns.append(cleaned_v4)
        seen = {}
        for ind , col in enumerate(clean_columns):
            if col not in seen :
                seen[col]= 0
            else :
                seen[col]+= 1
                clean_columns[ind]= col + "_" + f"{seen[col]}"

        df.columns = clean_columns



        # print(f"after the cleaning {dirty_df.columns}")
        return df
    clean_column_names_df = clean_column_names(df=dirty_df)
    # print(f"after the cleaning{clean_column_names_df.columns}")
    # print(clean_column_names_df.head())
    # print(clean_column_names_df.dtypes)


    def normalisation(df):
        df = df.replace(["N/A", "NA", "null", "None", "", "-","n/a", "#", "?", "--"], np.nan)
        # print(f"missing values for each column :\n{df.isna().sum()}")
        return df
    normalised_df = normalisation(df=clean_column_names_df)

    def Convert_Types(df):

        for col in df.columns:

            print(f"\n{col} : {df[col].dtype}")

            # Remember values that were already missing
            missing_before = df[col].isna()

            # ==================================================
            # 1. Try BOOLEAN
            # ==================================================

            boolean_values = {
                "true": True,
                "false": False,
                "yes": True,
                "no": False,
            }

            converted_boolean = (
                df[col]
                .astype(str)
                .str.lower()
                .map(boolean_values)
            )

            boolean_failed = (
                converted_boolean.isna() & ~missing_before
            )

            number_failed = boolean_failed.sum()

            number_tested = (~missing_before).sum()

            if number_tested > 0:
                success_rate = (
                    number_tested - number_failed
                ) / number_tested
            else:
                success_rate = 0

            if success_rate >= 0.95:
                df[col] = converted_boolean
                print("→ detected as boolean")
                continue

            # ==================================================
            # 2. Try NUMERIC
            # ==================================================

            converted_numeric = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            missing_after = converted_numeric.isna()

            conversion_failed = missing_after & ~missing_before

            number_failed = conversion_failed.sum()

            if number_tested > 0:
                success_rate = (
                    number_tested - number_failed
                ) / number_tested
            else:
                success_rate = 0

            if success_rate >= 0.95:

                # Check if all numeric values are whole numbers
                if (converted_numeric.dropna() % 1 == 0).all():
                    df[col] = converted_numeric.astype("Int64")
                    print("→ detected as integer")
                else:
                    df[col] = converted_numeric.astype("Float64")
                    print("→ detected as float")

                continue

            # ==================================================
            # 3. Try DATETIME (only if values look date-like)
            # ==================================================

            sample = df[col].dropna().astype(str).head(20)
            looks_like_date = sample.str.contains(r"[-/:]").mean() >= 0.5

            if looks_like_date:

                converted_datetime = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                missing_after = converted_datetime.isna()

                conversion_failed = missing_after & ~missing_before

                number_failed = conversion_failed.sum()

                if number_tested > 0:
                    success_rate = (
                        number_tested - number_failed
                    ) / number_tested
                else:
                    success_rate = 0

                if success_rate >= 0.95:
                    df[col] = converted_datetime
                    print("→ detected as datetime")
                    continue

            # ==================================================
            # 4. Nothing matched
            # ==================================================

            print("→ keeping as string/object")

        return df

    col_converted_types = Convert_Types(df=normalised_df)
    # print(col_converted_types.dtypes)


    ##### Duplicate Detection & Removal (AFTER type conversion) #####

    def Duplicate_Detection_Removal(df):

        number_duplicates_rows = df.duplicated().sum()
        # print(f"number is : {number_duplicates_rows}")


        cleaned_total_numb_rows = df.shape[0]
        percentage = (number_duplicates_rows / cleaned_total_numb_rows ) * 100
        # print(f"percentage :{round(percentage,2)}%")

        n_rows_befor = df.shape[0]
        # print(f"number of rows and columns befor Duplicate_Detection_Removal :\n{n_rows_befor}")
        cleaned_df = df.drop_duplicates(keep="first")
        n_rows_after = cleaned_df.shape[0]
        # print(f"number of rows and columns after Duplicate_Detection_Removal :\n{n_rows_after}")
        duplicate_metrics = {
                                "duplicates_found": int(df.duplicated().sum()),
                                "duplicates_percentage": float(round(percentage , 2)),
                                "rows_before": int(n_rows_befor),
                                "rows_after": int(n_rows_after),
                            }

        return (cleaned_df,duplicate_metrics)

    cleaned_from_dup_df, duplicate_metrics = Duplicate_Detection_Removal(df = col_converted_types )

    # print( cleaned_from_dup_df, duplicate_metrics)


    def handle_missing_values(df):
        numeric_count = 0
        non_numeric_count = 0
        count_per_column ={}


        for col in df.columns :
            if pd.api.types.is_numeric_dtype(df[col]) :
                median_ = df[col].median()
                v1 = int(df[col].isna().sum())
                numeric_count += v1
                count_per_column[col] = v1
                df[col] = df[col].fillna(median_)

            else :
                v2 = int(df[col].isna().sum())
                non_numeric_count += v2
                count_per_column[col]=v2
                df[col] = df[col].fillna("unknown")

        total = numeric_count+non_numeric_count
        missing_metrics = {"count_per_column" :count_per_column ,
                           "non_numeric_count" : int(non_numeric_count) ,
                           "numeric_count" : int(numeric_count) ,
                           "total" : int(total)
                           }

        return df , missing_metrics

    clean_df, missing_metrics = handle_missing_values(df=cleaned_from_dup_df)  # noqa: RUF059
    # print(clean_df.isna().sum())

    ##### detect anomalies #####
    CONTAMINATION = float(args.contamination)

    def detect_anomalies(df, contamination) :
        model = IsolationForest(contamination=contamination, random_state=42)

        only_num_col = df.select_dtypes(include="number")
        copy_df = only_num_col.fillna(only_num_col.median())

        if only_num_col.empty :
            print("warnning , there is no numeric columns ! ")
            return df , {}
        else :
            print("data has numeric data")
            copy_df = only_num_col.copy()

            prediction = model.fit_predict(copy_df)
            anomalies = prediction == -1
            anomalies_count = anomalies.sum()

            percentage = (anomalies_count / len(prediction) ) * 100
            metrics = {
                            "anomalies_found": int(anomalies.sum()),
                            "anomaly_percentage": float(round(percentage , 2)),
                            "contamination": float(contamination),
                            "numeric_columns_used": list(copy_df.columns),
                        }
            df["isolation_flag"] = anomalies
            return df,metrics

    df_with_flag , anomaly_metrics =detect_anomalies(df = clean_df , contamination=CONTAMINATION)


    ##### build_embeddings #####
    messy_columns = list(df_with_flag.columns)
    canonical_column_names_ = CANONICAL_FIELDS
    def build_embeddings(canonical_column_names ,messy_columns_):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        messy_embeddings = model.encode(messy_columns_)
        canonical_embeddings = model.encode(canonical_column_names)
        # print(messy_embeddings.shape)
        # print(canonical_embeddings.shape)
        similarities = model.similarity(messy_embeddings , canonical_embeddings)
        # print(similarities)
        return (model ,canonical_column_names ,canonical_embeddings)
    model_emb ,canonical_column_names_,canonical_embeddings_ = build_embeddings(canonical_column_names =canonical_column_names_ ,messy_columns_ = messy_columns )

    def preprocess_messy_column__name(name) :
        ABBREVIATIONS = {
                            "nm": "name",
                            "dt": "date",
                            "id": "identifier",
                            "dob": "date of birth",
                            "addr": "address",
                            "tel": "telephone",
                            "ph": "phone",
                            "qty": "quantity",
                            "amt": "amount",
                            "num": "number",
                            "no": "number",
                            "dept": "department",
                            "emp": "employee",
                            "cust": "customer",
                            "prod": "product",
                            "family":"last"
                        }
        v1 = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
        v2 = v1.replace("_"," ").replace("-"," ")
        v3 = re.sub(r"\s+", " ", v2).lower()

        if " " in v3 :
            splited_v3 = v3.split()

            for ab in ABBREVIATIONS :
                for index , item in enumerate(splited_v3) :
                    if ab == item :
                        splited_v3[index] = ABBREVIATIONS[ab]

            v3 = " ".join(splited_v3)
        else :
            for ab in ABBREVIATIONS :
                if ab == v3 :
                    v3 = ABBREVIATIONS[ab]
        # print(v3)
        return v3

    THRESHOLD=float(args.threshold)
    def match_columns(df ,model, canonical_names , canonical_embd , threshold):
        row_results = []
        cols_to_match = [c for c in df.columns if c != "isolation_flag"] #hena kan7iyed les columns likent zed o3arfthom dyalach
        for col in cols_to_match :

            pre_processed_col = preprocess_messy_column__name(col)
            # bdel had algorihtem method b batch processing o9ra 3lih
            messy_col_emb_1D = model.encode(pre_processed_col) # ghadi t3tik ghy vector fih (384,) number
            messy_col_emb_2D= np.expand_dims(messy_col_emb_1D, axis=0)


            similarity_number = cosine_similarity(canonical_embd , messy_col_emb_2D)

            # print(f"similarity is : {similarity_number}")
            # similarity is : [   [0.10335775]
            #                     [0.76258785]
            #                     [0.5800252 ]
            #                     [0.21306303]
            #                     [0.15669516]
            #                     [0.17364316]
            #                     [0.03854379]
            #                     [0.06067825]
            #                     [0.14743029]
            #                     [0.02737785]
            #                     [0.00870144] ]
            #                     isolaation flag
            #khasni ghi the best match , y3ni l max , onbiyen l colomn  messy , m3a chmen column canonical t matcha m3ah
            the_best_score = float(np.clip(similarity_number.max(), 0.0, 1.0))# kent dair flowel float(similarity_number.max()) hit l9it problem ki3tini 1.0011 , y3ni fo9 1
            # print(f"the best is : {col} {the_best_score}")

            #DABA 3NDI VALUE DYAL COLUMN CANONICAL , KHASNO SMIYA DYALO
            index = np.argmax(similarity_number) #bach njbed index
            matched_to = canonical_names[index]
            # print(f"the canonical column is : {matched_to}")
            dic = { "raw column":col ,
                    "matched_to" : matched_to ,
                    "confidence" : the_best_score}
            row_results.append(dic)
        # print(f"raw results :\n{row_results}")

        for dic in row_results :
            confidence = dic["confidence"]
            if confidence >= threshold :
                dic["is_confident"] = True
            else :
                dic["is_confident"] = False
        finale_result = row_results
        # print(len(finale_result))
        confident_TRUE_count = 0
        confident_FALSE_count = 0
        total_confidence = 0
        total = len(finale_result)

        for dic in finale_result :
            if dic["is_confident"] == True :
                confident_TRUE_count += 1
            if dic["is_confident"] == False :
                confident_FALSE_count += 1
            total_confidence += dic["confidence"]
        average_confidence = round(total_confidence / total, 3)

        metrics = {"total_columns" : len(finale_result),
                   "columns_mapped_confident" : confident_TRUE_count ,
                   "columns_low_confidence" : confident_FALSE_count ,
                   "average_confidence" : average_confidence}
        # print(metrics)

        # print(finale_result)
        return (finale_result , metrics)


    finale_result ,match_metrics = match_columns(df = df_with_flag ,model = model_emb, canonical_names = canonical_column_names_ ,canonical_embd= canonical_embeddings_, threshold=THRESHOLD)
    # print(f"finale result {finale_result} \n match matrics{match_metrics}")



    ##### Apply the Renames + Write the Review File #####
    def write_review_file(df , results ,path):
            to_df = {}
            list_row_column = []
            is_confident_false_results =[i for i in results if i["is_confident"] == False]
            # print(is_confident_false_results)
            if not is_confident_false_results :
                empty_dic = {"review_file_written": False, "review_columns": []}
                return empty_dic
            else :
                for i in is_confident_false_results :
                    raw_col = i["raw column"]
                    to_df[raw_col]= df[i["raw column"]]
                    list_row_column.append(raw_col)
                    # print(df[i["raw column"]])
                    # print(type(df[i["raw column"]]))
            # print(to_df)
            df = pd.DataFrame(to_df)
            review_file_written = True
            review_file_path = path
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try :
                df.to_csv(path, index=False)

            except OSError as e :
                print(f"error :\n{e}")
                review_file_written = False
                review_file_path = None
                list_row_column = []
            dic ={
                "review_file_written": review_file_written,
                "review_file_path": review_file_path,
                "review_columns": list_row_column,
            }
            # print("the dict :\n",dic )
            return dic
    review_metrics =write_review_file(df=df_with_flag, results=finale_result , path = "review/review_these_rows.csv")

    def renaming(df,results):
        rename_count_sucs =0
        skipped_due_to_collision = 0
        left_for_review = 0
        already_correct = 0
        columns = list(df.columns)
        rename_dic =  {}
        for i in results:
            if not i["is_confident"]:
                left_for_review += 1
                i["final_status"] = "low_confidence"
                continue

            raw = i["raw column"]
            target = i["matched_to"]

            if raw == target:
                # already correctly named
                already_correct += 1
                i["final_status"] = "already_correct"
            elif target not in columns and target not in rename_dic.values():
                # safe to rename
                rename_dic[raw] = target
                rename_count_sucs += 1
                i["final_status"] = "renamed"
            else:
                # real collision: target already exists (in original cols or in planned renames)
                skipped_due_to_collision += 1
                i["final_status"] = "skipped_collision"
        df = df.rename(columns=rename_dic)

        rename_matrics = {
                                "renames_applied": rename_count_sucs,
                                "renames_skipped_collision": skipped_due_to_collision,
                                "columns_left_unmapped": left_for_review,
                                "already_correct":already_correct
                            }




        return df ,rename_matrics
         #(renamed_df, rename_metrics)

    rename_df , rename_metrics = renaming(df = df_with_flag , results=finale_result)

    # print(rename_df.columns , "\n" , rename_metrics)


    def save_cleaned_csv(finale_df,finale_result_path):
        df_shap = finale_df.shape
        rows_= df_shap[0]
        columns_ = df_shap[1]

        os.makedirs(os.path.dirname(finale_result_path), exist_ok=True)
        cleaned_file_written = True
        cleaned_file_path = finale_result_path
        try :
            finale_df.to_csv(finale_result_path, index=False)

        except OSError as e :
            print(f"error :\n{e}")
            cleaned_file_written = False
            cleaned_file_path = None

        dic={
            "cleaned_file_written": cleaned_file_written,
            "cleaned_file_path":cleaned_file_path ,
            "rows_written": rows_,
            "columns_written": columns_,
            }
        # print(dic)
        return dic


    cleaned_metrics = save_cleaned_csv(finale_df=rename_df , finale_result_path="finale_resulte/cleaned_csv.csv")

    def collect_metrics_to_context():
        all_the_metrics = {
            "input_file":input_file ,
            "run_time": run_date.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time": round(total_time , 2),
            "rows": dirty_df_shape[0],
            "columns": dirty_df_shape[1],
            "missing_metrics": missing_metrics,
            "duplicate_metrics": duplicate_metrics,
            "anomaly_metrics": anomaly_metrics,
            "match_metrics": match_metrics,
            "rename_metrics": rename_metrics,
            "review_metrics": review_metrics,
            "cleaned_metrics": cleaned_metrics,
            "mapping_rows": finale_result,  # the list of dicts
            "threshold": THRESHOLD,
            }
        return all_the_metrics
    def html_report(context,path):
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("report_template.html")
        html = template.render(context)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        webbrowser.open(path)








    ##### timer stops #####
    end_ = time.perf_counter()
    total_time = end_ - start_
    # print(f"it takes {total_time}s")
    context_ = collect_metrics_to_context()
    html_report(context=context_ , path="report/data_report.html")
    print(context_)
if __name__ == "__main__" :
    main()


    