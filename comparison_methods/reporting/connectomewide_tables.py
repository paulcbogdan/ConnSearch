import os
import pathlib

import pandas as pd
from scipy import stats as stats

from ConnSearch import utils
from ConnSearch.reporting.ConnSearch_tables import get_idx_to_label
from comparison_methods.reporting.connectomewide_xl import create_xl_table


def make_network_pair_table(top_rank_edges, fp_csv, coords,
                            search_closest=False,
                            atlas='power', do_open=True):
    '''
    Creates a table representing the prevalence of top edges for connectome-wide
        methods. Edges are organized base don network-network pairs (e.g.,
        DAN-SM). This function was used to make Table 2.
    The p-values associated with network pairs generated here are based on
        binomial distributions. However, more accurate p-values can be achieved
        with permutation-testing, then added in manually. Future work
        may incorporate this automatically.
    The same table is represented in three ways:
        (1) .csv containing this info
        (2) .xlsx file containing this info and formatted nicely
        (3) .xlsx file in 2 but now only looking at pairs of networks where
            weighted prevalence surpasses 1.0. Table 2 is based on this.
    '''

    # idx_to_label maps each ROI to its corresponding Yeo atlas network label
    root_dir = pathlib.Path(__file__).parent.parent.parent
    fp_idx_to_label = fr'{root_dir}/pickle_cache/' \
                      f'search{search_closest}_{atlas}_idx_to_label_b.pkl'
    idx_to_label = utils.pickle_wrap(fp_idx_to_label,  # cache hastens future runs
                                     lambda: get_idx_to_label(coords,
                                                              search_closest=search_closest),
                                     easy_override=True)
    # relative_frequency is a dict mapping each network to its number of ROIs
    relative_frequency = get_node_distribution_stats(idx_to_label)
    relative_frequency_pairs = {}  # will map network pairs to number of edges
    intra_total = 0
    inter_total = 0
    for network0, freq0 in relative_frequency.items():
        for network1, freq1 in relative_frequency.items():
            # Expected number of edges between a given pair of networks can be
            #   calculated based on the number of ROIs in each network
            if network0 == network1:  # intra-network corresponds to (n*(n-1))/2
                relative_frequency_pair = (freq0) * (freq0 - 1) / 2
                intra_total += relative_frequency_pair
            else:  # inter-network corresponds to n1*n2
                relative_frequency_pair = (freq0) * (freq1)
                if network0 < network1:
                    inter_total += relative_frequency_pair
            relative_frequency_pairs[(network0, network1)] = int(relative_frequency_pair)

    cnter_intra, cnter_inter = get_cntr(top_rank_edges, idx_to_label)
    n_edges = len(top_rank_edges)
    df_intra = organize_networkpair_df(cnter_intra, relative_frequency_pairs,
                                       intra_total + inter_total, n_top=n_edges)
    assert df_intra['#Features'].sum() == intra_total, \
        f'{df_intra["#Features"].sum()=} but {intra_total=}'
    print('\n                        ***  Intra-network table  ***')
    print(df_intra)
    print('\n                        ***  Inter-network table  ***')
    df_inter = organize_networkpair_df(cnter_inter, relative_frequency_pairs,
                                       inter_total + intra_total, n_top=n_edges)
    assert df_inter['#Features'].sum() == inter_total

    print(df_inter)
    print(f'num top {n_edges} are internetwork: {sum(df_inter[f"Top {n_edges}"])}')

    fp_xl = fp_csv.replace('.csv', '_xl.xlsx')
    create_xl_table(df_intra, df_inter, fp_xl)

    fp_xl = fp_csv.replace('.csv', '_xl_above1.xlsx')
    create_xl_table(df_intra[df_intra['Weighted Prevalence'] > 1],
                    df_inter[df_inter['Weighted Prevalence'] > 1], fp_xl)

    df = pd.concat([df_intra, df_inter])

    df['Weighted Prevalence'] = df['Weighted Prevalence'].apply('{:.2f}'.format)
    df['p-value'] = df['p-value'].apply(lambda x: f'{x:.3f}' if x > .001 else '.001')
    print('Number of features total:', df['#Features'].sum())
    print(f'Connectome-wide table: {fp_csv=}')
    try:
        df.to_csv(fp_csv, index=False)
        if do_open:
            os.system(f'open {fp_csv}')
    except PermissionError as e:
        print('Can\'t save table, already open:')
        print(f'\t{e}')


def get_node_distribution_stats(idx_to_label):
    '''
    Counts number of ROIs associated with each network and returns a dict
        mapping network labels to counts.
    :param idx_to_label: dict, mapping each ROI to its corresponding Yeo atlas
                               network label
    :return: dict
    '''
    regions = idx_to_label.values()
    cnt = {}
    [cnt.update({region: cnt.get(region, 0) + 1}) for region in regions]
    return cnt


def organize_networkpair_df(cnter, relative_frequency_pairs, total, n_top=1000):
    '''
    Creates a dataframe representing info about top edges between network pairs
    '''
    df_as_l = []
    for network_pair, cnt in cnter.items():
        rel_freq = relative_frequency_pairs[network_pair]
        if rel_freq == 0:
            weight = 0
        elif network_pair[0] == network_pair[1]:
            weight = total / rel_freq
        elif network_pair[1] > network_pair[0]:  # compare alphabetically
            continue
        else:
            weight = total / rel_freq
        p_val = get_binomial_p_value(cnt, rel_freq, total, n_top=n_top)
        df_as_l.append({'Network Pair': '-'.join(network_pair),
                        '#Features': rel_freq,
                        f'Top {n_top}': cnt,
                        'Weighted Prevalence': cnt * weight / n_top,
                        'p-value': p_val})

    df = pd.DataFrame(df_as_l)
    df.sort_values('Weighted Prevalence', inplace=True, ascending=False)
    return df


def get_binomial_p_value(n_top_edges_network, n_edges_network, n_edges_connectome,
                         n_top=1000):
    prob_at_most = stats.binom.cdf(k=n_top_edges_network, n=n_top,
                                   p=n_edges_network / n_edges_connectome)
    return 1 - prob_at_most


def get_cntr(top_rank_edges, idx_to_label):
    networks = list(set(idx_to_label.values()))
    cnter_inter = {}
    cnter_intra = {}

    for i_network in networks:
        for j_network in networks:
            if i_network == j_network:
                cnter_intra[(i_network, j_network)] = 0
            else:
                cnter_inter[(i_network, j_network)] = 0
                cnter_inter[(j_network, i_network)] = 0

    for ij in top_rank_edges:
        i_network = idx_to_label[ij[0]]
        j_network = idx_to_label[ij[1]]
        if i_network == j_network:
            cnter_intra[(i_network, j_network)] += 1
        else:
            cnter_inter[(i_network, j_network)] += 1
            cnter_inter[(j_network, i_network)] += 1

    return cnter_intra, cnter_inter
