import pandas as pd

att = pd.read_excel('TCMSP_test_herb_gloabal_att.xlsx', na_filter=False, index_col=0)
score = pd.read_excel('edge_pred_tcmsp.xlsx', na_filter=False, index_col=0)

molecule_target = pd.read_excel('test_message_molecule_target.xlsx', na_filter=False)
target_disease = pd.read_excel('test_message_target_disease.xlsx', na_filter=False)


def find_pairs(alist, blist, pairs_list):
    found_pairs = []
    for a in alist:
        for b in blist:
            if [a, b] in pairs_list:
                found_pairs.append([a, b])
            if [b, a] in pairs_list:
                found_pairs.append([b, a])
    return found_pairs


top_k = 100
for index, row in att.iterrows():
    top_k_set = row.sort_values(ascending=False).head(top_k)  # 排序并取前1

    attention_entity = list(top_k_set.index)

    # print(top_k_set)
    top_k_set = pd.DataFrame([top_k_set.values], index=[index], columns=top_k_set.index)

    if index in score.index:
        row = score.loc[index]
        top_k_pred = row[row > 0.001].sort_values(ascending=False)

        predict_entity =list(top_k_pred.index)

        # print(top_k_pred)
        top_k_pred = pd.DataFrame([top_k_pred.values], index=[index], columns=top_k_pred.index)

        top_k_pred = top_k_pred.reset_index()
        top_k_pred = pd.DataFrame([top_k_pred.columns.tolist()] + top_k_pred.values.tolist())

        top_k_set = top_k_set.reset_index()
        top_k_set = pd.DataFrame([top_k_set.columns.tolist()] + top_k_set.values.tolist())

        new_df = []
        new1 = find_pairs(alist=attention_entity, blist=predict_entity, pairs_list=molecule_target.values.tolist())
        new2 = find_pairs(alist=attention_entity, blist=predict_entity, pairs_list=target_disease.values.tolist())
        new_df += new1 + new2
        new_df = pd.DataFrame(new_df)

        if len(new_df.values)>0:
            print(index)
            pred_score = pd.concat([top_k_set, top_k_pred, new_df], axis=0)
            pred_score.to_excel(index + '.xlsx', index=False)
        else:
            print('no')
            pred_score = pd.concat([top_k_set, top_k_pred], axis=0)
            pred_score.to_excel(index + '.xlsx', index=False)


    print()
