import pandas as pd
import json
from CausaLM.Reviews_Features.lm_finetune import pregenerate_training_data, topics_finetune_on_pregenerated
from CausaLM.constants import REVIEWS_FEATURES, REVIEWS_FEATURES_TREAT_CONTROL_MAP_FILE, REVIEWS_FEATURES_DATASETS_DIR

CAUSAL_GRAPH_TRETEMENT_COL = 'col_name'
CAUSAL_GRAPH_CONTROL_COL = 'control'

def create_json_map(treated_feature, control_features):
    # with open(REVIEWS_FEATURES_TREAT_CONTROL_MAP_FILE, "r") as jsonfile:
    #     reviews_features_treat_dict = json.load(jsonfile)

    # reviews_features_treat_dict[treated_feature]['treated_feature'] = treated_feature
    # reviews_features_treat_dict[treated_feature]['control_features'] = [control_features]

    reviews_features_treat_dict = {treated_feature: {'treated_feature': treated_feature,
                                   'control_features': [control_features]}}

    with open(REVIEWS_FEATURES_TREAT_CONTROL_MAP_FILE, 'w') as fp:
        json.dump(reviews_features_treat_dict, fp)


if __name__ == '__main__':
    path = f"{REVIEWS_FEATURES_DATASETS_DIR}/causal_graph.csv"
    causal_graph_df = pd.read_csv(path)
    for index, row in causal_graph_df.iterrows():
        treatment = row[CAUSAL_GRAPH_TRETEMENT_COL]
        control = row[CAUSAL_GRAPH_CONTROL_COL]
        create_json_map(treatment, control)
        pregenerate_training_data.main(treatment)
        topics_finetune_on_pregenerated.main(treatment)





