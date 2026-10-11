import json  # noqa: I001
import re
from canonical_column_names import CANONICAL_FIELDS
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# functions
canonical_column_names_ = CANONICAL_FIELDS

def build_embeddings(canonical_column_names):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        canonical_embeddings = model.encode(canonical_column_names)
        return model ,canonical_embeddings
model_emb ,canonical_embeddings_ = build_embeddings(canonical_column_names =canonical_column_names_ )
# print(canonical_embeddings_)
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
        v3 = re.sub(r"[^a-zA-Z0-9 ]", "", v2)
        v4 = re.sub(r"\s+", " ", v3).lower()

        if " " in v4 :
            splited_v4 = v4.split()

            for ab in ABBREVIATIONS :
                for index , item in enumerate(splited_v4) :
                    if ab == item :
                        splited_v4[index] = ABBREVIATIONS[ab] 

            v4 = " ".join(splited_v4)
        else :
            for ab in ABBREVIATIONS :
                if ab == v4 :
                    v4 = ABBREVIATIONS[ab] 
        # print(v4)          
        return v4





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
        
        list_name_score = []
        for idx in top_3 :
            name = canonical_names[idx]
            score = float(np.clip(similarity[idx], 0.0, 1.0))
            # 0.3394421339035034 exp score
            name_score= (name , round(score, 3))
            # ('status', 0.33)

            list_name_score.append(name_score)
        final_result.append(list_name_score)
    # print(final_result)
    return final_result
def evaluator(path_for_json_matcher , model, canonical_embeddings, canonical_names, threshold):
    total_count_of_everything = 0
    top1_count =0
    top3_count = 0 
    silent_errors = 0
    flagged = 0
    failures = []
    total_per_category={}

    with open(path_for_json_matcher, mode="r", encoding="utf-8") as f:
        list_raw_data = []
        data = json.load(f)
        # print(data) 
        for i in data :
            list_raw_data.append(i["raw"])
        # print(list_raw_data)
     
    list_of_name_score = top3_predect(model=model , canonical_embeddings = canonical_embeddings ,
                                       canonical_names=canonical_names , raw_name=list_raw_data)
    # print(list_of_name_score)
    # list of these :
    # [('remote_work', 0.41), ('employee_id', 0.242), ('email', 0.241) , .....]

    for case , predect_name_score in zip(data,list_of_name_score):
        total_count_of_everything += 1  # noqa: SIM113

        top1_name , top1_score = predect_name_score[0]
        # 3lach dert predect_name_score[0] bach njbed men hado 
        # [('remote_work', 0.41), ('employee_id', 0.242), ('email', 0.241)]
        # ghit lwel : ('remote_work', 0.41) 
        # kola mra 

        correct = case["correct"]
        category = case["category"]


         

        if category not in total_per_category :
            total_per_category[category]= {"total":0,
                                             "top1_count":0}
            
        total_per_category[category]["total"]+= 1
        #exemple total_per_category = {
        #                               "category":EXEMPLE "casing_space"{
        #                               "correct": 5 ,kola mra l9a dik la categ ghaizid wahed bach nhseb total 
        #                              }
        #                 }
        
        if top1_name == correct :
            top1_count += 1
            total_per_category[category]["top1_count"]+=1

        
        else :
            
            if top1_score >= threshold:
                    silent_errors += 1      
            else:
                flagged += 1
                dictionary = {"raw": case["raw"],
                                            "predicted": top1_name,
                                            "correct": correct,
                                            "score": top1_score,
                                            "category": category}
                                
                failures.append(dictionary)

        for i in predect_name_score:
            if i[0] == correct:
                top3_count += 1
                break
                
    # print(failures)
    result = {
        "total": total_count_of_everything,
        "top1_accuracy": top1_count / total_count_of_everything,
        "top3_accuracy": top3_count / total_count_of_everything,
        "silent_errors": silent_errors,
        "flagged": flagged,
        "per_category": total_per_category,
        "failures": failures,
        }
    print(f"\nTest set size: {result['total']}\n")
    print(f"Top-1 accuracy:     {top1_count}/{total_count_of_everything}  ({100*result['top1_accuracy']:.1f}%)")
    print(f"Top-3 accuracy:     {top3_count}/{total_count_of_everything}  ({100*result['top3_accuracy']:.1f}%)")
    print(f"Silent errors:      {silent_errors}/{total_count_of_everything}  ({100*silent_errors/total_count_of_everything:.1f}%)")
    print(f"Flagged for review: {flagged}/{total_count_of_everything}  ({100*flagged/total_count_of_everything:.1f}%)")

    print("\nBreakdown by category:")
    for cat in sorted(total_per_category):
        c = total_per_category[cat]
        pct = 100 * c["top1_count"] / c["total"]
        print(f"  {cat:<15} {c['top1_count']}/{c['total']}  ({pct:.1f}%)")

    print("\nFailures (top-1 wrong):")
    for f in failures[:20]:
        print(f"  [{f['category']:<15}] \"{f['raw']}\" → \"{f['predicted']}\" ({f['score']:.2f}) expected: \"{f['correct']}\"")





         
    
evaluator( path_for_json_matcher = "test_data_for_matcher.json" ,model=model_emb , 
          canonical_embeddings = canonical_embeddings_ , canonical_names = canonical_column_names_, threshold=0.75)
        


            
    
            
             

        


             
        

     
    
