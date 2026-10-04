import json
import re
from canonical_column_names import CANONICAL_FIELDS
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# functions
canonical_column_names_ = CANONICAL_FIELDS

def build_embeddings(canonical_column_names):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        canonical_embeddings = model.encode(canonical_column_names)
        return model ,canonical_embeddings
model_emb ,canonical_embeddings_ = build_embeddings(canonical_column_names =canonical_column_names_ )
print(canonical_embeddings_)
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





def top3_predect(model, canonical_embeddings, canonical_names, raw_name):
    final_result = []
    for raw in raw_name :

        clean_name_col = preprocess_messy_column__name(raw)
        # print(clean_name_col)
        embed_messy_col_1D = model.encode(clean_name_col)
        embed_messy_col_2D = np.expand_dims(embed_messy_col_1D , axis = 0)

        similarity = cosine_similarity(canonical_embeddings , embed_messy_col_2D).ravel()
        top_3 = np.argsort(-similarity)[:3]
        # print("top3 :\n",top_3)
        # top3 :
        #     [10  0  8]

        list_n_s = []
        for idx in top_3 :
            name = canonical_names[idx]
            score = float(np.clip(similarity[idx], 0.0, 1.0))
            # 0.3394421339035034 exp score
            n_s= (name , round(score, 3))
            # ('status', 0.33)

            list_n_s.append(n_s)
        final_result.append(list_n_s)
    # print(final_result)
    return final_result
def evaluator(path_for_json_matcher , model, canonical_embeddings, canonical_names, threshold):
    with open(path_for_json_matcher, mode="r", encoding="utf-8") as f:
        list_raw_data = []
        data = json.load(f)
        for i in data :
            list_raw_data.append(i["raw"])
        # print(list_raw_data)

    list_of_n_s = top3_predect(model=model , canonical_embeddings = canonical_embeddings , canonical_names=canonical_names , raw_name=list_raw_data)
    print(list_of_n_s)

evaluator( path_for_json_matcher = "test_data_for_matcher.json" ,model=model_emb , canonical_embeddings = canonical_embeddings_ , canonical_names = canonical_column_names_, threshold=0.75)
















