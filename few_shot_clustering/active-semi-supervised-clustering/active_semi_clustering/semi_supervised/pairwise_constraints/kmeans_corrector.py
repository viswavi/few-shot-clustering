from collections import defaultdict
import copy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm

from active_semi_clustering.active.pairwise_constraints import ExampleOracle
from active_semi_clustering.active.pairwise_constraints.example_oracle import MaximumQueriesExceeded

class KMeansCorrection:
    def __init__(self, oracle, cluster_predictions, cluster_centers, labels):
        self.oracle = oracle
        self.cluster_predictions = cluster_predictions
        self.cluster_centers = cluster_centers
        self.labels = labels

    @staticmethod
    def normalize_features(features):
            kg_features = normalize(features[:, :300], axis=1, norm="l2")
            bert_features = normalize(features[:, 300:], axis=1, norm="l2")
            return np.hstack([kg_features, bert_features])

    def fit(self, X, num_corrections):
        features = self.normalize_features(X)
        point_cluster_distances = euclidean_distances(features, self.cluster_centers)

        closest = np.min(point_cluster_distances, axis=1)
        second_closest = np.partition(point_cluster_distances, 1, axis=1)[:, 1]
        top_two_gap = second_closest - closest
        closest_top_two = np.argsort(top_two_gap)[:num_corrections]
        oracle = self.oracle

        ambiguous_points_cluster_distances = point_cluster_distances[closest_top_two]

        ambiguous_points_cluster_rankings = np.argsort(ambiguous_points_cluster_distances, axis=1)
        ambiguous_points_top_five_clust_idxs = ambiguous_points_cluster_rankings[:, :5]

        inertias = []
        inertias_by_cluster = defaultdict(list)
        for i, l in enumerate(self.cluster_predictions):
            inertias.append(point_cluster_distances[i, l])
            inertias_by_cluster[l].append((i, point_cluster_distances[i, l]))
        representative_points = {}
        for l in inertias_by_cluster:
            members_sorted = sorted(inertias_by_cluster[l], key=lambda x: x[1])
            least_inertia_points = [m[0] for m in members_sorted[:3]]
            representative_points[l] = least_inertia_points

        corrected_labels = copy.deepcopy(self.cluster_predictions)

        example_oracle = ExampleOracle(self.labels, max_queries_cnt=100000)
        top_cluster_labels = []
        for i, ent_idx in tqdm(enumerate(closest_top_two)):
            correct_found = False
            top_clust = ambiguous_points_top_five_clust_idxs[i][0]
            top_cluster_rep_points = representative_points[top_clust]
            top_cluster_bad = False
            for top_cluster_rep in top_cluster_rep_points:
                top_cluster_class = example_oracle.query(ent_idx, top_cluster_rep)
                if top_cluster_class is False:
                    top_cluster_bad = True
                    break
            if not top_cluster_bad:
                top_cluster_labels.append(0)
                continue
            for i, next_best_clust in enumerate(ambiguous_points_top_five_clust_idxs[i][1:]):
                next_best_cluster_rep_points = representative_points[next_best_clust]
                make_correction = False
                for next_best_cluster_rep in next_best_cluster_rep_points:
                    next_cluster_class = example_oracle.query(ent_idx, next_best_cluster_rep) 
                    if top_cluster_bad and next_cluster_class is True:
                        make_correction = True
                        break
                if make_correction:
                    top_cluster_labels.append(i+1)
                    correct_found = True
                    break
            if not correct_found:
                top_cluster_labels.append(-1)

                
        top_cluster_predictions = []
        for i, ent_idx in tqdm(enumerate(closest_top_two)):
            correct_found = False
            top_clust = ambiguous_points_top_five_clust_idxs[i][0]
            top_cluster_rep_points = representative_points[top_clust]
            top_cluster_bad = False
            for top_cluster_rep in top_cluster_rep_points:
                top_cluster_class = self.oracle.query(ent_idx, top_cluster_rep)
                if top_cluster_class is False:
                    top_cluster_bad = True
                    break
            if not top_cluster_bad:
                top_cluster_predictions.append(0)
                continue
            for i, next_best_clust in enumerate(ambiguous_points_top_five_clust_idxs[i][1:]):
                next_best_cluster_rep_points = representative_points[next_best_clust]
                make_correction = False
                for next_best_cluster_rep in next_best_cluster_rep_points:
                    next_cluster_class = self.oracle.query(ent_idx, next_best_cluster_rep) 
                    if top_cluster_bad and next_cluster_class is True:
                        make_correction = True
                        break
                if make_correction:
                    top_cluster_predictions.append(i+1)
                    correct_found = True
                    break
            if not correct_found:
                top_cluster_predictions.append(-1)
                    
        num_gpt_corrections_false_positive =  len([x for i, x in enumerate(top_cluster_predictions) if top_cluster_predictions[i] > 0 and top_cluster_labels[i] == 0 ]); num_gpt_corrections_false_negative =  len([x for i, x in enumerate(top_cluster_predictions) if top_cluster_predictions[i] == 0 and top_cluster_labels[i] > 0 ]); num_gpt_corrections_true_positive =  len([x for i, x in enumerate(top_cluster_predictions) if top_cluster_predictions[i] > 0 and top_cluster_labels[i] == top_cluster_predictions[i] ]); num_gpt_corrections_true_negative =  len([x for i, x in enumerate(top_cluster_predictions) if top_cluster_predictions[i] == 0 and top_cluster_labels[i] == top_cluster_predictions[i] ])


        num_gpt_corrections_false_positive =  len([x for i, x in enumerate(top_cluster_predictions) if top_cluster_predictions[i] > 0 and top_cluster_labels[i] == 0 ])
        num_gpt_corrections_false_negative =  len([x for i, x in enumerate(top_cluster_predictions) if top_cluster_predictions[i] == 0 and top_cluster_labels[i] > 0 ])
        num_gpt_corrections_true_positive =  len([x for i, x in enumerate(top_cluster_predictions) if top_cluster_predictions[i] > 0 and top_cluster_labels[i] == top_cluster_predictions[i] ])
        num_gpt_corrections_true_negative =  len([x for i, x in enumerate(top_cluster_predictions) if top_cluster_predictions[i] == 0 and top_cluster_labels[i] == top_cluster_predictions[i] ])

        print(f"num_gpt_corrections_false_positive: {num_gpt_corrections_false_positive}")
        print(f"num_gpt_corrections_false_negative: {num_gpt_corrections_false_negative}")
        print(f"num_gpt_corrections_true_positive: {num_gpt_corrections_true_positive}")
        print(f"num_gpt_corrections_true_negative: {num_gpt_corrections_true_negative}")


        kmeans_corrections = []

        num_corrections_made = 0
        num_correct_corrections_made = 0
        num_queries = 0
        try:
            for i, ent_idx in tqdm(enumerate(closest_top_two)):
                top_clust = ambiguous_points_top_five_clust_idxs[i][0]
                top_cluster_rep_points = representative_points[top_clust]
                top_cluster_bad = False
                for top_cluster_rep in top_cluster_rep_points:
                    top_cluster_class = oracle.query(ent_idx, top_cluster_rep)
                    num_queries += 1
                    if top_cluster_class is False:
                        top_cluster_bad = True
                        break
                for next_best_clust in ambiguous_points_top_five_clust_idxs[i][1:]:
                    next_best_cluster_rep_points = representative_points[next_best_clust]
                    make_correction = False
                    for next_best_cluster_rep in next_best_cluster_rep_points:
                        next_cluster_class = oracle.query(ent_idx, next_best_cluster_rep) 
                        num_queries += 1
                        if top_cluster_bad and next_cluster_class is True:
                            make_correction = True
                            break

                    gt_make_correction = False
                    for next_best_cluster_rep in next_best_cluster_rep_points:
                        next_cluster_class = example_oracle.query(ent_idx, next_best_cluster_rep) 
                        num_queries += 1
                        if top_cluster_bad and next_cluster_class is True:
                            gt_make_correction = True
                            break

                    if make_correction == True:
                        num_corrections_made += 1
                        if gt_make_correction:
                            num_correct_corrections_made += 1
                        kmeans_corrections.append({"ent_idx": int(ent_idx), 
                                                   "previous_label": int(corrected_labels[ent_idx]),
                                                   "corrected_label": int(next_best_clust)})
                        corrected_labels[ent_idx] = next_best_clust
                        
                        break
        except MaximumQueriesExceeded:
            pass

        self.labels_ = corrected_labels

        breakpoint()

        print(f"Num oracle queries: {num_queries}")
        print(f"Num Corrections: {num_corrections_made}")
        print(f"Num Correct Corrections: {num_correct_corrections_made}")

        '''
        from active_semi_clustering.active.pairwise_constraints import ExampleOracle
        oracle = ExampleOracle(labels, max_queries_cnt=100000)

        from dataloaders import load_dataset, generate_synthetic_data
        import sys
        sys.path.append("cmvc")
        from cmvc.helper import invertDic
        from cmvc.metrics import pairwiseMetric, calcF1
        from cmvc.test_performance import cluster_test
        X, labels, side_information = load_dataset("OPIEC59k", "/projects/ogma1/vijayv/okb-canonicalization/clustering/data", "test")

        current_metrics = cluster_test(side_information.p, side_information.side_info, self.cluster_predictions, side_information.true_ent2clust, side_information.true_clust2ent)
        new_metrics = cluster_test(side_information.p, side_information.side_info, corrected_labels, side_information.true_ent2clust, side_information.true_clust2ent)

        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, macro_f1, micro_f1, pairwise_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons  = cluster_test(side_information.p, side_information.side_info, self.cluster_predictions, side_information.true_ent2clust, side_information.true_clust2ent)    
        '''
        return self