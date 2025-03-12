import matplotlib.pyplot as plt
import numpy as np
from vnc_networks.vnc_networks import params
from vnc_networks.vnc_networks.connections import Connections
import pandas
import matplotlib.patches as patche



def plot_cluster_similarity_matrix(vnc : Connections,subset : list[params.UID] = None , clustering_method : str = 'markov'
                                   ,distance_metric : str = 'cosine_in'
                                   ,cutoff : float = 0.5 ,c_min : int = 4):
    m = vnc.get_cmatrix()
    premotor_neurons = m.list_upstream_neurons(subset)
    nodes = list(set(subset).union(premotor_neurons))
    m.restrict_nodes(nodes)
    (
        clustered_cmatrix,  # clustered similarity matrix as cmatrix object
        uid_clusters,  # list of lists of uids in each cluster
        index_clusters,  # list of lists of indices in each cluster matching the clustered cmatrix
    ) = m.detect_clusters(
        distance=distance_metric,
        method=clustering_method,
        cutoff=cutoff,
        cluster_size_cutoff=c_min,
        cluster_data_type="uid",
        cluster_on_subset=subset,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    # Visualise the similarity matrix and its clusters
    clustered_sim_mat = clustered_cmatrix.get_matrix().todense()
    # create a matrix of zeros
    mat = np.zeros((clustered_sim_mat.shape[0], clustered_sim_mat.shape[1]))
    # draw the boundaries between clusters
    for cluster in index_clusters:
        mat[cluster[0]: cluster[-1] + 1, cluster[0]: cluster[-1] + 1] = 1
    _ = clustered_cmatrix.imshow(savefig=False, ax=ax, title="Clustered similarity matrix")
    ax.imshow(mat, cmap="binary", alpha=0.2)
    plt.show()


def show_data_connections(subnetwork : Connections) :

    def get_attributes_list (body_id : int) :
        uid = subnetwork.get_uids_from_bodyid(body_id)
        name = subnetwork.get_node_attribute(uid,"name")
        class_1 = subnetwork.get_node_attribute(uid,"class_1")
        return name,class_1

    def add_new_row(data_frame: pandas.DataFrame, source: str,source_name : str, source_class : str,
                    target: str,target_name : str, target_class : str, syn_count: int, eff_weights: int):
        new_row = {'source_body_ids': source,'source_name' : source_name, 'source_class' : source_class
            , 'target_body_id': target, 'target_name' : target_name, 'target_class' : target_class,
                   'syn_count': syn_count, 'eff_weights': eff_weights}
        data_frame.loc[len(data_frame)] = new_row

    data = subnetwork.get_dataframe()
    data_frame = pandas.DataFrame(columns=["source_body_ids","source_name","source_class", "target_body_id",
                                           "target_name","target_class", "syn_count", "eff_weights"])
    for index, row in data.iterrows():
        source_name, source_class = get_attributes_list(row["start_bid"])
        target_name, target_class = get_attributes_list(row["end_bid"])
        add_new_row(data_frame, row["start_bid"],source_name,source_class, row["end_bid"],target_name,target_class,
                    row["syn_count"], row["eff_weight"])

    display(data_frame)


def show_data_connections2(network : Connections, neurons_list : list[params.BodyId]) :
    data = network.get_dataframe()

    def get_attributes (uid : int) :
        name = network.get_node_attribute(uid,"name")
        class_1 = network.get_node_attribute(uid,"class_1")
        return name,class_1

    def add_new_row(data_frame: pandas.DataFrame, source: str,
                    source_bid : str ,source_name : str, source_class : str,
                    target: str, target_bid : str,target_name : str, target_class : str, syn_count: int, eff_weight: int):

        new_row = {'source_uid': source, 'source_bid' : source_bid, 'source_name' : source_name, 'source_class' : source_class
            , 'target_uid': target,'target_bid' : target_bid, 'target_name' : target_name, 'target_class' : target_class,
                   'syn_count': syn_count,"eff_weight" : eff_weight}
        data_frame.loc[len(data_frame)] = new_row

    data_frame = pandas.DataFrame(columns=["source_uid",'source_bid',"source_name", "source_class", "target_uid",'target_bid',
                                           "target_name", "target_class", "syn_count","eff_weight"])
    for source in neurons_list :
        downstream_neurons = network.get_neurons_downstream_of(source)
        source_name, source_class = get_attributes(source)
        source_bid = network.get_bodyids_from_uids(source)[0]
        for target in downstream_neurons :
            target_name, target_class = get_attributes(target)
            target_bid = network.get_bodyids_from_uids(target)[0]
            syn_count = network.get_nb_synapses(source,target)
            eff_weight = (data.loc[(data['start_bid'] == source_bid) & (data['end_bid'] == target_bid)]).iloc[0]["eff_weight"]
            add_new_row(data_frame,source,source_bid,source_name,source_class,target,target_bid
                        ,target_name,target_class,syn_count,eff_weight)
    return data_frame

def get_dataset_first_connections():
    return pandas.read_csv("dataset.csv")

def get_syn_info(df,name : str):
    ds = df.loc[df['source_name'] == name]
    return ds["syn_count"].mean(), ds["syn_count"].std()

def get_weight_info(df,name : str):
    ds = df.loc[df['source_name'] == name]
    return ds["eff_weight"].mean(), ds["eff_weight"].std()

def get_nb_neurons_target(df,source_name : str, target_class : str):
    ds = df.loc[(df['source_name'] == source_name) & (df['target_class'] == target_class)]
    return len(ds)

def get_nb_syn_target(df,source_name : str, target_class : str):
    ds = df.loc[(df['source_name'] == source_name) & (df['target_class'] == target_class)]
    s = ds["syn_count"].sum()
    return s

def plot_repartition(df,name : str):
    target_list = target_class_list
    nb_list = []
    nb_syn_list = []
    for target_class in target_list:
        nb_list.append(get_nb_neurons_target(df,name,target_class))
        nb_syn_list.append(get_nb_syn_target(df,name,target_class))
    data = pandas.DataFrame({"Number of neurons" : nb_list, "Number of synapses" : nb_syn_list}, index = target_list)
    data.plot.pie(subplots = True,legend = False,autopct='%1.1f%%',figsize =(15,20))

def plot_target_count(df, min_nb : int = 10):
    def color_plot(df):
        data_color = []
        for name in df["target_class"]:
            if name == 'motor':
                data_color.append('red')
            elif name == 'ascending':
                data_color.append('orange')
            elif name == 'intrinsic':
                data_color.append('cyan')
            elif name == 'descending':
                data_color.append('blue')
            elif name == 'efferent_ascending':
                data_color.append('darkred')
            elif name == 'sensory':
                data_color.append('lime')
            elif name == 'sensory_ascending':
                data_color.append('springgreen')
            elif name == 'efferent':
                data_color.append('yellow')
            elif (name == 'interneuron_unknown' or name == 'unknown'):
                data_color.append('dimgray')
            elif name == 'glia':
                data_color.append('purple')
            else:
                data_color.append('black')
        return data_color

    data = df["target_name"].value_counts()
    index = np.where(data < min_nb)
    data = data[:index[0][0]]
    color = color_plot(df)

    p_m = patche.Patch(color='red', label='Motor')
    p_a = patche.Patch(color='orange', label='Ascending')
    p_i = patche.Patch(color='cyan', label='intrinsic')
    p_d = patche.Patch(color='blue', label='descending')
    p_ef = patche.Patch(color='darkred', label='efferent_ascending')
    p_s = patche.Patch(color='lime', label='sensory_ascending')
    p_e = patche.Patch(color='yellow', label='efferent')
    p_u = patche.Patch(color='dimgray', label='unknown')
    p_g = patche.Patch(color='purple', label='glia')
    p = data.plot(
        kind='bar',
        figsize=(70, 70),
        color=color
    )
    plt.xticks([])
    p.legend(handles=[p_m, p_a, p_e, p_i, p_d, p_ef, p_s, p_u, p_g])
    return p

def plot_source_class_info(df, syn: bool = True):
    l = list_name
    info_list = []
    for name in l:
        if (syn):
            mean, std = get_syn_info(df, name)
        else:
            mean, std = get_weight_info(df, name)

        entry = {"Name": name, "Weight_mean": mean}
        info_list.append(entry)
    data = pandas.DataFrame.from_dict(dic).set_index('Name')
    return data.plot(
        kind='bar',
        ylabel="Mean",
        xlabel='Name',
        figsize=(70, 70),
        color='purple'
    )
list_name =['DNxn128', 'DNxn089', 'DNut054', 'DNht001', 'DNxn175', 'DNxn160', 'DNut032',
 'DNxl085', 'DNxn017', 'DNxl057', 'DNut045', 'DNxn187', 'DNfl003', 'DNfl028',
 'DNnt007', 'DNxl059', 'DNut007', 'DNlt008', 'DNxl046', 'DNxn104', 'DNxl003',
 'DNxn085', 'DNxn117', 'DNut026', 'DNxl071', 'DNxn151', 'DNut015', 'DNxn107',
 'DNxn046', 'DNnt018', 'DNxl056', 'DNxn039', 'DNxn064', 'DNxl027', 'DNxn024',
 'DNxn014', 'DNxn103', 'DNxn013', 'DNxn137', 'DNxn171', 'DNfl039', 'DNxl102',
 'DNut044', 'DNit007', 'DNxl075', 'DNxl035', 'DNxn095', 'DNxn144', 'DNxn041',
 'DNut029', 'DNfl030', 'DNxn012', 'DNxl055', 'DNxl050', 'DNxl135', 'DNut011',
 'DNfl026', 'DNxl068', 'DNut033', 'DNut047', 'DNnt011', 'DNxl040', 'DNut006',
 'DNlt002', 'DNxl104', 'DNut028', 'DNxl074', 'DNut042', 'DNxn062', 'DNxn105',
 'DNfl033', 'DNxl082', 'DNfl010', 'DNlt011', 'DNit006', 'DNxn172', 'DNxn116',
 'DNxl086', 'DNxn061', 'DNxl045', 'DNfl015', 'DNnt002', 'DNit008', 'DNxn180',
 'DNfl032', 'DNlt009', 'DNad001', 'DNxn019', 'DNut013', 'DNxl103', 'DNxn048',
 'DNxn139', 'DNxn173', 'DNxn140', 'DNut005', 'DNut036', 'DNxl093', 'DNxn109',
 'DNxl016', 'DNxl107', 'DNxn115', 'DNxn114', 'DNxl126', 'DNxn124', 'DNut018',
 'DNxn169', 'DNxl116', 'DNnt021', 'DNxl115', 'DNxn179', 'DNxl002', 'DNfl021',
 'DNxn020', 'DNxn148', 'DNxl095', 'DNnt009', 'DNxl029', 'DNxl105', 'DNnt012',
 'DNxn145', 'DNad003', 'DNxn100', 'DNxl028', 'DNut016', 'DNxl013', 'DNxl114',
 'DNxn015', 'DNfl040', 'DNxn156', 'DNnt014', 'DNnt003', 'DNfl001', 'DNfl011',
 'DNxn023', 'DNxn004', 'DNxn167', 'DNxn147', 'DNut009', 'DNxl010', 'DNxn044',
 'DNxn091', 'DNxl054', 'DNxl128', 'DNxl043', 'DNxl051', 'DNut046', 'DNxn055',
 'DNfl012', 'DNxl110', 'DNfl038', 'DNxn136', 'DNxn186', 'DNut055', 'DNxl120',
 'DNxl032', 'DNxn126', 'DNxn143', 'DNut056', 'DNxl111', 'DNxn060', 'DNlt006',
 'DNnt005', 'DNfl025', 'DNxn152', 'DNxl038', 'DNxl127', 'DNxn098', 'DNxl099',
 'DNxn181', 'DNxn071', 'DNut051', 'DNlt003', 'DNxn108', 'DNut004', 'DNut038',
 'DNxn040', 'DNxl030', 'DNut043', 'DNxl122', 'DNxl125', 'DNfl017', 'DNxn097',
 'DNnt008', 'DNfl037', 'DNxn158', 'DNfl042', 'DNxn133', 'DNxn113', 'DNxl090',
 'DNxl008', 'DNut012', 'DNxn028', 'DNxl001', 'DNnt010', 'DNxl024', 'DNfl016',
 'DNxn031', 'DNut023', 'DNxn106', 'DNut039', 'DNxl063', 'DNxl049', 'DNit003',
 'DNad005', 'DNut008', 'DNxn042', 'DNxl131', 'DNxn112', 'DNut037', 'DNxn066',
 'DNxl020', 'DNxn058', 'DNxn110', 'DNxl096', 'DNxl087', 'DNxn079', 'DNxn130',
 'DNfl002', 'DNxn072', 'DNxn150', 'DNxn022', 'DNut019', 'DNxl067', 'DNxl014',
 'DNut053', 'DNlt010', 'DNxn070', 'DNxn174', 'DNxl037', 'DNnt017', 'DNut041',
 'DNxl118', 'DNxn065', 'DNfl004', 'DNxn161', 'DNxn076', 'DNxl005', 'DNxn155',
 'DNxn068', 'DNxn088', 'DNxn123', 'DNut061', 'DNxl041', 'DNut014', 'DNxl094',
 'DNit001', 'DNfl006', 'DNxn036', 'DNxn163', 'DNxn182', 'DNxl072', 'DNxl044',
 'DNxn121', 'DNnt013', 'DNxn075', 'DNxn049', 'DNxl053', 'DNxl033', 'DNxl006',
 'DNxn035', 'DNhl001', 'DNxl007', 'DNxn125', 'DNnt016', 'DNxn054', 'DNxl117',
 'DNxn094', 'DNfl045', 'DNlt005', 'DNut052', 'DNxn077', 'DNnt006', 'DNnt019',
 'DNxn129', 'DNxn168', 'DNxn011', 'DNut003', 'DNxn093', 'DNxl079', 'DNfl035',
 'DNxl062', 'DNxn032', 'DNxn176', 'DNxl066', 'DNxl052', 'DNut010', 'DNut021',
 'DNxl004', 'DNut057', 'DNxn057', 'DNut017', 'DNxl065', 'DNfl007', 'DNlt001',
 'DNxl123', 'DNxn073', 'DNxn010', 'DNxn069', 'DNxl113', 'DNxl080', 'DNxl047',
 'DNxl018', 'DNxn183', 'DNxl026', 'DNxl021', 'DNfl019', 'DNit004', 'DNxl100',
 'DNxn029', 'DNxn052', 'DNut024', 'DNxn067', 'DNxn021', 'DNxn159', 'DNxl039',
 'DNxn016', 'DNxn138', 'DNut022', 'DNfl041', 'DNxl101', 'DNxl009', 'DNxl091',
 'DNxl025', 'DNxl124', 'DNxn122', 'DNxl077', 'DNfl014', 'DNxn135', 'DNxn006',
 'DNut058', 'DNut060', 'DNut001', 'DNwt001', 'DNxn059', 'DNut031', 'DNad004',
 'DNut002', 'DNut050', 'DNxl098', 'DNxl012', 'DNxl129', 'DNfl036', 'DNxn142',
 'DNxl058', 'DNnt004', 'DNfl020', 'DNxn025', 'DNxn164', 'DNxl019', 'DNxn131',
 'DNml001', 'DNxl092', 'DNxn149', 'DNut048', 'DNxn056', 'DNxl112', 'DNxn166',
 'DNxl132', 'DNxn082', 'DNxl042', 'DNxn005', 'DNxn146', 'DNxl031', 'DNxl034',
 'DNit005', 'DNxl084', 'DNxn192', 'DNxl083', 'DNxn102', 'DNxn051', 'DNfl034',
 'DNfl023', 'DNut040', 'DNxl097', 'DNxl073', 'DNxn083', 'DNxn101', 'DNut059',
 'DNxn080', 'DNxl121', 'DNxn096', 'DNxn162', 'DNnt001', 'DNxl133', 'DNit002',
 'DNxl078', 'DNfl022', 'DNxn074', 'DNxn087', 'DNxl011', 'DNxn170', 'DNfl024',
 'DNxn178', 'DNfl029', 'DNxl061', 'DNxn111', 'DNxn078', 'DNxn001', 'DNxn045',
 'DNfl027', 'DNut034', 'DNad002', 'DNxn053', 'DNxl070', 'DNxn018', 'DNxn118',
 'DNxn127', 'DNxn092', 'DNxl048', 'DNxn007', 'DNxn141', 'DNxl119', 'DNxn033',
 'DNxn119', 'DNxn099', 'DNfl018', 'DNxn008', 'DNfl009', 'DNxn037', 'DNxn081',
 'DNut049', 'DNxn026', 'DNxn184', 'DNxl060', 'DNxn009', 'DNut025', 'DNut035',
 'DNxn165', 'DNxn002', 'DNxl015', 'DNxl017', 'DNlt007', 'DNxl081', 'DNxn189',
 'DNxn084', 'DNxn043', 'DNxn090', 'DNxn177', 'DNxn132', 'DNut030', 'DNxn050',
 'DNxn185', 'DNfl008', 'DNxl106', 'DNxl088', 'DNlt004', 'DNfl031', 'DNxl134',
 'DNxl109', 'DNxn134', 'DNfl043', 'DNxn047', 'DNnt015', 'DNxn157', 'DNxl023',
 'DNxn030', 'DNxl036', 'DNxn027', 'DNxn034', 'DNit010', 'DNnt022', 'DNit011',
 'DNxl076', 'DNit013', 'DNfl044', 'DNxn063', 'DNnt020', 'DNfl005', 'DNut027',
 'DNxn190', 'DNfl013', 'DNut020', 'DNxl022', 'DNad006', 'DNxl108', 'DNut062',
 'DNxn154', 'DNxn191', 'DNxn188', 'DNxn038', 'DNxn193', 'DNxn003', 'DNxl064',
 'DNxl130', 'DNxn153', 'DNxn086', 'DNit012', 'DNxl069', 'DNxn120', 'DNit009']

target_class_list = ['motor', 'ascending', 'intrinsic', 'descending', 'efferent_ascending',
 'sensory', 'efferent', 'sensory_ascending', 'interneuron_unknown', 'unknown',
 'glia']
