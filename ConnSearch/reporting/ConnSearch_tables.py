import os
import pickle

import pandas as pd
import pathlib

from ConnSearch import utils
from nichord import get_idx_to_label, find_closest

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 120)


def get_region_distribution(pkg, idx_to_label):
    '''
    Used for generating the "Network Component Composition" columns of Tables 1
        & 3. Counts the numbers of ROIs in a component within each
        network, following the labels in idx_to_label.
    :param pkg: dict, contains key 'components' which is a list of ROI indices
    :param idx_to_label: dict, maps ROI index to network label
    :return: dict, mapping network label to an int (the count)
    '''
    node_indices = [idx for idx in pkg['component']]
    networks = list(map(idx_to_label.get, node_indices))
    networks = list(map(lambda x: x.replace('_', ' '), networks))
    cnt = {}
    for network in networks:
        cnt.update({network: cnt.get(network, 0) + 1})
    cnt = {k: v / len(pkg['component']) for k, v in cnt.items()}
    return cnt


def get_neuromorphometrics_center(coord, shorthand=True):
    '''
    Used for generating the "Region" and "BA" columns of Tables 1 & 3. It
        also helps with creating the table caption. It returns name of the
        region corresponding to a coordinate using the neuromoephometrics atlas.
        Also returns the Brodmann area, based on the Talairach atlas. It also
        returns other helpful info for generating the legend caption.
    For example, get_neuromorphometrics_center((20,40,20), shorthand=True)
        returns: ('R MFG', 'R MFG, Middle Frontal Gyrus', 'MFG, Middle Frontal Gyrus', '9')

    :param coord: tuple, (x, y, z) coordinate in MNI space
    :param shorthand: if True, returns a shorthand name for the region
    :return:
    '''
    # The coord's region or the closest region within 10 mm is used.
    #   find_closest returns 'uncertain' if no region is found within 10 mm.
    region_str, dist = find_closest('neuromorphometrics', coord, max_dist=10)
    region = '_'.join(region_str.split('_')[:2])
    region = \
        region.replace('Right_', 'R ').replace('Left_', 'L ').replace('_', ' ')
    # key is used for the table caption, e.g., 'R MFG, Middle Frontal Gyrus'
    key = region + ', ' + ' '.join(map(lambda x: x.capitalize(),
                                       region_str.split('_')[2:]))
    # key_no_hemi removes L/R, e.g., 'MFG, Middle Frontal Gyrus'
    key_no_hemi = region.replace('R ', '').replace('L ', '') + ', ' + ' '.join(
        map(lambda x: x.capitalize(), region_str.split('_')[2:]))
    if 'Cerebellum' in region:
        BA = ''
    else:
        BA, dist = find_closest('talairach_ba', coord, must_have=['Brodmann'], max_dist=10)
        BA = BA.replace('Brodmann_area_', '')
    if not shorthand:
        region = region_str
    return region, key, key_no_hemi, BA


def generate_component_table(fp_csv, dir_results, search_closest=True,
                             open_csv=False, score_name='Accuracy',
                             atlas='power'):
    '''
    Generates the ConnSearch tables. Used for Tables 1 & 3. Operates by loading
        each .pkl result from dir_results and plotting each result ("pkg") as
        a row.
    :param fp_csv: str, filepath to save table as .csv
    :param dir_results: str, directory containing .pkl results
    :param search_closest: bool, if True, uses the closest network label within
        5 mm for ROIs which don't fall cleanly within any network
    :param open_csv: bool, if True, opens the .csv file after saving
    :param score_name: str, name of the score column
    :param atlas: str, name of the atlas used for the analysis
    :return:
    '''
    fns = os.listdir(dir_results)
    assert len(fns) > 0, 'No significant results found.'
    pkgs = []
    for fn in fns:
        fp = os.path.join(dir_results, fn)
        with open(fp, 'rb') as f:
            pkg = pickle.load(f)
            pkgs.append(pkg)
    root_dir = pathlib.Path(__file__).parent.parent.parent
    fp_idx_to_label = fr'{root_dir}/pickle_cache/' \
                       f'search{search_closest}_{atlas}_idx_to_label.pkl'
    idx_to_label = utils.pickle_wrap(fp_idx_to_label,# cache hastens future runs
                                     lambda: get_idx_to_label(pkg['coords_all'],
                                            search_closest=search_closest),
                                     easy_override=False)
    df_as_l = []
    keys = []

    # Get all the information needed for the table and the table caption
    for entry_n, pkg in enumerate(pkgs):
        core_coord = pkg['coords_all'][pkg['component'][0]]
        core_region, key, key_no_hemi, BA = get_neuromorphometrics_center(core_coord)
        if pkg['score'] < 1:
            score = pkg['score']*100
        else:
            score = pkg['score']
        table_row = {'Core': core_region, 'BA': BA, score_name: score,
                     'ROI #': pkg['component'][0], 'y': core_coord[1]}
        table_row.update(get_region_distribution(pkg, idx_to_label))
        df_as_l.append(table_row)
        keys.append(key_no_hemi)

    # Format the pandas dataframe nicely
    df = pd.DataFrame(df_as_l)
    df.sort_values(by=['y'], inplace=True) # sort Core ROI anterior to posterior
    df.drop(columns=['y'], inplace=True)
    df.fillna(0, inplace=True)
    df = df.round(decimals=3)
    df['Entry #'] = list(range(1, len(df)+1))
    excl = ['Entry #', 'Core', 'BA', 'ROI #', score_name]
    labels_sans_center = list(set(df.columns) - set(excl))
    percent_cols = labels_sans_center + [score_name] if score_name == 'accuracy' else labels_sans_center
    df[percent_cols] = df[percent_cols].applymap(lambda x: f'{x:.0%}')
    df = df.reindex(excl + sorted(labels_sans_center), axis=1)

    # Prepare the key part of the table caption,
    #  eg, table_caption = "AnG, Angular gyrus; ITG, Inferior temporal gyrus..."
    already_in = set()
    keys_pruned = []
    for key in keys:
        if key in already_in: continue
        keys_pruned.append(key)
        already_in.add(key)
    table_caption = '; '.join(keys_pruned)
    fp_caption = os.path.join(os.path.dirname(fp_csv),
                              'caption_' + \
                              os.path.basename(fp_csv).replace('.csv', '.txt'))
    with open(fp_caption, 'w') as f:
        f.write(table_caption) # Writes the caption keys to a .txt
    print(f'{table_caption=}') # Print caption keys to console

    print(f'Save table to: {fp_csv=}')
    try:
        df.to_csv(fp_csv, index=False)
        if open_csv:
            os.system(f'open {fp_csv}')
    except PermissionError as e:
        print(f'Can\'t save table, already open: {e}')

if __name__ == '__main__':
    print(get_neuromorphometrics_center((-16, -77, 34)))