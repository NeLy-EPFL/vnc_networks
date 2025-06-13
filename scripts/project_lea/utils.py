import matplotlib.pyplot as plt
import numpy as np
from vnc_networks.vnc_networks import params
from vnc_networks.vnc_networks.connections import Connections
import pandas
import matplotlib.patches as patche
import math
import scipy as sc
import copy

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


def show_data_connections_intrinsic_to_motor(network : Connections, neurons_list : list[params.BodyId]) :
    data = network.get_dataframe()

    def get_attributes (uid : int) :
        name = network.get_node_attribute(uid,"name")
        class_1 = network.get_node_attribute(uid,"class_1")
        type = network.get_node_attribute(uid,"nt_type")
        target = network.get_node_attribute(uid,"target")
        return name,class_1,type,target

    def add_new_row(data_frame: pandas.DataFrame, source: str,
                    source_bid : str ,source_name : str, source_class : str, source_type : str,
                    target: str, target_bid : str,target_name : str, target_class : str, target_type : str,target_muscle : str, syn_count: int, eff_weight: int):

        new_row = {'source_uid': source, 'source_bid' : source_bid, 'source_name' : source_name, 'source_class' : source_class, 'source_type' : source_type
            , 'target_uid': target,'target_bid' : target_bid, 'target_name' : target_name, 'target_class' : target_class, 'target_type' : target_type,
                   'target_muscle' : target_muscle,'syn_count': syn_count,"eff_weight" : eff_weight}
        data_frame.loc[len(data_frame)] = new_row

    data_frame = pandas.DataFrame(columns=["source_uid",'source_bid',"source_name", "source_class","source_type", "target_uid",'target_bid',
                                           "target_name", "target_class","target_type","target_muscle", "syn_count","eff_weight"])
    for source in neurons_list :
        downstream_neurons = network.get_neurons_downstream_of(source)
        source_name, source_class, source_type, _ = get_attributes(source)
        source_bid = network.get_bodyids_from_uids(source)[0]
        for target in downstream_neurons :
            target_name, target_class, target_type, target_muscle = get_attributes(target)
            if target_class == 'motor':
                target_bid = network.get_bodyids_from_uids(target)[0]
                syn_count = network.get_nb_synapses(source,target)
                eff_weight = (data.loc[(data['start_bid'] == source_bid) & (data['end_bid'] == target_bid)]).iloc[0]["eff_weight"]
                add_new_row(data_frame,source,source_bid,source_name,source_class,source_type,target,target_bid
                            ,target_name,target_class,target_type,target_muscle,syn_count,eff_weight)
    return data_frame

def get_dataset_first_connections():
    return pandas.read_csv("dataset.csv")

def get_syn_info(df,name : str):
    ds = df.loc[df['source_name'] == name]
    return ds["syn_count"].mean(), ds["syn_count"].std()

def get_weight_info(df,name : str):
    ds = df.loc[df['source_name'] == name]
    return ds["eff_weight"].mean()

def get_nb_neurons_target(df,source_name : str, target_class : str):
    ds = df.loc[(df['source_name'] == source_name) & (df['target_class'] == target_class)]
    return len(ds)

def get_nb_syn_target(df,source_name : str, target_class : str):
    ds = df.loc[(df['source_name'] == source_name) & (df['target_class'] == target_class)]
    s = ds["syn_count"].sum()
    return s

def plot_source_syn_mean_or_std(df,type : str = 'mean', sort_type : str = 'none'):
    l = list_name
    data_list = []
    for name in l:
        mean, std = get_syn_info(df, name)
        if(type == 'mean'):
            entry = {"Name": name, "syn_mean": mean}
            color = 'blue'
            label = "syn_mean"
        else :
            entry = {"Name": name, "syn_std": mean}
            color = 'orange'
            label = 'syn_std'
        data_list.append(entry)

    def sort_by_mean(e):
        return e["syn_mean"]
    def sort_by_std(e):
        return e["syn_std"]
    if(sort_type == 'mean'):
        data_list.sort(key=sort_by_mean)
    elif (sort_type == 'std'):
        data_list.sort(key = sort_by_std)

    data = pandas.DataFrame.from_dict(data_list).set_index('Name')
    p = data.plot(
        kind='bar',
        ylabel=label,
        xlabel='Name',
        figsize=(70, 70),
        color=color
    )
    return p


def plot_weight_distribution(df, sort_type : str = 'none'):
    l = list_name
    data_list = []
    for name in l:
        mean = get_weight_info(df, name)
        entry = {"Name": name, "weight_mean": mean}
        data_list.append(entry)
    def sort_by_mean(e):
        return e["weight_mean"]

    if(sort_type == 'mean'):
        data_list.sort(key=sort_by_mean)
    data = pandas.DataFrame.from_dict(data_list).set_index('Name')
    p = data.plot(
        kind='bar',
        ylabel='weight_mean',
        xlabel='Name',
        figsize=(70, 70),
        color='purple'
    )
    return p

def plot_ratio_syn_per_class(df,name : str):
    target_list = target_class_list
    list_disparity = []
    for target_class in target_list:
        if get_nb_neurons_target(df,name,target_class)==0:
            list_disparity.append(None)
        else:
            list_disparity.append((get_nb_syn_target(df,name,target_class)/get_nb_neurons_target(df,name,target_class)))
    data = pandas.DataFrame({"Ratio synapses per neurons" : list_disparity}, index = target_list)
    data.plot.bar()

def plot_repartition(df,name : str):
    target_list = target_class_list
    nb_list = []
    nb_syn_list = []
    for target_class in target_list:
        nb_list.append(get_nb_neurons_target(df,name,target_class))
        nb_syn_list.append(get_nb_syn_target(df,name,target_class))
    data = pandas.DataFrame({"Number of neurons" : nb_list, "Number of synapses" : nb_syn_list}, index = target_list)
    data.plot.pie(subplots = True,legend = False,autopct='%1.1f%%',figsize =(15,20))

def plot_target_count(df, upper_bound : int = -1, lower_bound : int = -1, show_name : bool = False):
    data = df["target_name"].value_counts()

    def color_plot(df, data):
        data_color = []
        for name in data:
            index = np.where(df["target_name"] == name)[0][0]
            target_class = df["target_class"][index]
            if target_class == 'motor':
                data_color.append('red')
            elif target_class == 'ascending':
                data_color.append('darkorange')
            elif target_class == 'intrinsic':
                data_color.append('green')
            elif target_class == 'descending':
                data_color.append('blue')
            elif target_class == 'efferent_ascending':
                data_color.append('khaki')
            elif target_class == 'sensory':
                data_color.append('lime')
            elif target_class == 'sensory_ascending':
                data_color.append('springgreen')
            elif target_class == 'efferent':
                data_color.append('yellow')
            elif (target_class == 'interneuron_unknown' or target_class == 'unknown'):
                data_color.append('dimgray')
            elif target_class == 'glia':
                data_color.append('purple')
            else:
                data_color.append('black')
        return data_color

    if (upper_bound != -1) and (lower_bound != -1):
        index = np.where((data <= upper_bound) & (data > lower_bound))[0]
        data = data[index[0]: index[-1]]
    elif (upper_bound != -1) and (lower_bound == -1):
        index = np.where((data <= upper_bound))[0]
        data = data[index[0]:]
    elif (upper_bound == -1) and (lower_bound != -1):
        index = np.where((data > lower_bound))[0]
        data = data[:index[-1]]

    color = color_plot(df, data.index)

    p_m = patche.Patch(color='red', label='Motor')
    p_a = patche.Patch(color='darkorange', label='Ascending')
    p_i = patche.Patch(color='green', label='intrinsic')
    p_d = patche.Patch(color='blue', label='descending')
    p_ef = patche.Patch(color='khaki', label='efferent_ascending')
    p_s = patche.Patch(color='lime', label='sensory_ascending')
    p_e = patche.Patch(color='yellow', label='efferent')
    p_u = patche.Patch(color='dimgray', label='unknown')
    p_g = patche.Patch(color='purple', label='glia')
    p = data.plot(
        kind='bar',
        figsize=(70, 70),
        color=color
    )
    if (not show_name):
        plt.xticks([])
    p.legend(handles=[p_m, p_a, p_e, p_s, p_i, p_d, p_ef, p_u, p_g])
    return p

def plot_cluster_distribution(connections_dataset, cluster_dataset,brain_to_VNC_dataset,file_name : str
                              , old_dataset : bool = True):
    cluster_list = np.zeros(13)
    labels = ["No cluster", "Anterior grooming", "Take-off/Landing", "Walking", "Flight", "5", "6", "7", "8", "9",
              "Steering", "Flight", "12"]
    if old_dataset :
        column_name_vnc = 'VNC_type'
        column_name_brain = 'Brain_type'
    else :
        column_name_vnc = 'MANC DN name'
        column_name_brain = 'FAFB DN name'

    for index, row in connections_dataset.iterrows():
        name = row['source_name']
        brain_name = brain_to_VNC_dataset.loc[(brain_to_VNC_dataset[column_name_vnc] == name)][column_name_brain].iloc[0]
        info = cluster_dataset.loc[(cluster_dataset['DN name'] == brain_name)]
        print(info)
        cluster_list[info['Cluster number in figure']] += 1
    mask = np.where(cluster_list > 0, True, False)
    cluster_list_filtered = cluster_list[mask]
    labels_filtered = np.array(labels)[mask]
    fig, ax = plt.subplots()
    ax.pie(cluster_list_filtered[1:], autopct='%1.1f%%', labels=labels_filtered[1:])
    plt.title("Distribution of where the DNs comes from for T3 neuropil")
    plt.savefig(file_name)

# Plot the distribution of target neurons for every neurons in neuron_list

def plot_repartition_list(df, neuron_list, path_name : str = None):
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:pink', 'gold', 'tab:purple', 'tab:olive',
              'tab:gray', 'tab:gray', 'tab:gray']
    target_list = np.array(target_class_list)
    # Define subplot grid size
    n = len(neuron_list)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    for i,type in enumerate(neuron_list):
        l = []
        for target_class in target_list:
            l.append(get_nb_neurons_target(df, type, target_class))
        mask = np.where(np.array(l) > 2, True, False)
        l_filtered = np.array(l)[mask]
        target_list_filtered = target_list[mask]
        _,_,m = axes[i].pie(l_filtered, labels=target_list_filtered, autopct='%1.1f%%',colors = np.array(colors)[mask])
        axes[i].set_title(type)
        axes[i].set_facecolor('white')
        [m[j].set_color('white') for j in range(len(m))]

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if path_name is not None:
        plt.savefig(path_name)
    plt.show()

# Show how many synapses are used to target avery class in the VNC, one plot for every neuron in the list
def plot_ratio_syn_per_class_list(df, neuron_list, path_name : str = None):
    target_list = target_class_list
    dic={}
    for type in neuron_list :
        list_ratio = []
        for target_class in target_list:
            if get_nb_neurons_target(df,type,target_class)==0:
                list_ratio.append(None)
            else:
                list_ratio.append((get_nb_syn_target(df,type,target_class)/get_nb_neurons_target(df,type,target_class)))
        dic[type] = list_ratio
    data = pandas.DataFrame(dic, index = target_list)
    data.plot.bar(subplots = True, layout= (4,3),figsize = (10,10))
    if path_name is not None:
        plt.savefig(path_name)
    plt.show()

# Compare the average number of synapses and std between neurons
def plot_avg_synapse(df, neuron_list):
    syn_mean = []
    syn_std = []
    for type in neuron_list :
        m,s = get_syn_info(df,type)
        syn_mean.append(m)
        syn_std.append(s)
    data = pandas.DataFrame({"synapses mean" : syn_mean,"synapses std" : syn_std}, index = neuron_list)
    data.plot.bar(figsize = (8,8))
    plt.savefig("t_synapses")



def plot_cluster_distribution_new(connections_dataset, cluster_dataset,brain_to_VNC_dataset,file_name : str = None
                              ,suptitle : str = None, old_dataset : bool = True, get_distribution : bool = False):
    cluster_list = np.zeros(15)
    labels = ["No cluster", "Anterior grooming", "Take-off/Landing", "Walking", " 4 Flight", "5", "6", "7", "8","Steering", "10 Flight", "11", "12",'No matching to brain',"Brain name not found in cluster data"]
    colors = ['C7','C0','C1','C2','C3','C4','C5','C6','blueviolet','C8','C9','orangered','indigo','darkgrey','lightgrey']
    if old_dataset :
        column_name_vnc = 'VNC_type'
        column_name_brain = 'Brain_type'
    else :
        column_name_vnc = 'MANC DN name'
        column_name_brain = 'FAFB DN name'

    for index, row in connections_dataset.iterrows():
        name = row['source_name']
        brain_name = brain_to_VNC_dataset.loc[(brain_to_VNC_dataset[column_name_vnc] == name)][column_name_brain]
        if not brain_name.empty :
            not_found = True
            for _, brain in brain_name.items():
                if type(brain) == float: #NaN so didn't found a name
                    cluster_list[13] +=1
                else :
                    info = cluster_dataset.loc[(cluster_dataset['DN name'] == brain)]
                    if not info.empty:
                        c_list = []
                        for index_c, row_c in info.iterrows():
                            cluster = row_c['Cluster number in figure']
                            c_list.append(int(cluster))
                            not_found = False
                        for e in set(c_list):
                            cluster_list[e]+=1
                        break
            if not_found :
                cluster_list[14] +=1
        else :
            cluster_list[13] += 1
    if get_distribution :
        return cluster_list[1:13]
    else :
        mask = np.where(cluster_list > 0, True, False)
        cluster_list_filtered = cluster_list[mask]
        labels_filtered = np.array(labels)[mask]
        colors_filtered = np.array(colors)[mask]
        fig, ax = plt.subplots()
        ax.pie(cluster_list_filtered, autopct='%1.1f%%', labels=labels_filtered,colors = colors_filtered)
        plt.title("Distribution of where the DNs comes from")
        if file_name is not None:
            plt.savefig(file_name)


def create_cluster_data(list_name,dataset_intrinsic,data_cluster,brain_to_vnc_dataset,vnc):
    labels = ["1","2","3","4","5","6","7","8","9","10","11","12","label","max","color","nt_type",'uid']
    df_cluster = []

    for name in list_name:
        cluster_distribution, nt_type, label, max, cluster_color, nt_color, uids = get_label_intrinsic(name,dataset_intrinsic,data_cluster,brain_to_vnc_dataset, vnc)
        cluster_distribution = np.append(cluster_distribution,[label,max,cluster_color,nt_type,uids[0]])

        data_cluster_dic=dict(zip(labels,cluster_distribution),index = [name])
        df_cluster.append(pandas.DataFrame(data_cluster_dic))
    df_merged = pandas.concat(df_cluster)
    df_cluster_no_index = df_merged[
        [
            "1","2","3","4","5","6","7","8","9","10","11","12"
        ]
    ].values
    return df_merged, df_cluster_no_index

def get_label_intrinsic(name : str,dataset_brain_inter,data_cluster,brain_to_vnc_dataset,vnc):

    data = dataset_brain_inter.loc[(dataset_brain_inter['target_name'] == name)]
    if not data.empty :
        uids = list(data["target_uid"])
        color_scheme = ['C7','C0','C1','C2','C3','C4','C5','C6','blueviolet','C8','C9','orangered','indigo']
        nt_type = vnc.get_node_attribute(uids[0],'nt_type')
        cluster_distribution = plot_cluster_distribution_new(data,data_cluster,brain_to_vnc_dataset,old_dataset=False, get_distribution=True)
        max_idx = np.argmax(cluster_distribution)
        max = cluster_distribution[max_idx]
        label  = max_idx + 1 if max > 0 else 0
        match nt_type:
                case 'gaba':
                    nt_color = 'blue'
                case 'glutamate':
                    nt_color = 'violet'
                case 'acetylcholine':
                    nt_color = 'darkorange'
                case _ :
                   nt_color = 'C7'
        cluster_color = color_scheme[label]
        return cluster_distribution,nt_type,label,max,cluster_color,nt_color,uids
    else:
        return [], None, None, 0, 'C7', 'C7', [-1]


def get_data_distribution_motor(name : str, df_inter_motor, keys : list):
    dict_motor = dict.fromkeys(keys,0)
    motor_data = df_inter_motor.loc[(df_inter_motor['source_name'] == name)]
    motor_n_list = list(motor_data['target_name'])
    for mn in motor_n_list:
        dict_motor[mn] += 1
    return list(dict_motor.values())

def get_data_distribution_muscle(name : str, df_inter_motor, muscle_keys : list):
    dict_muscle = dict.fromkeys(muscle_keys,0)
    muscle_data = df_inter_motor.loc[(df_inter_motor['source_name'] == name)]
    muscle_n_list = list(muscle_data['target_muscle'])
    for mn in muscle_n_list:
        if mn not in muscle_keys:
            dict_muscle[len(muscle_keys)] += 1
        else :
            dict_muscle[mn] +=1
    return list(dict_muscle.values())

def create_inter_motor_data(list_name : list, df_inter_motor, dataset_brain_inter, data_cluster, brain_to_vnc, vnc, keys : list, motor : bool = True, muscle_keys=None):
    if muscle_keys is None:
        muscle_keys = []
    data = []
    for name in list_name:
        if motor:
            distribution = get_data_distribution_motor(name,df_inter_motor,keys)
            labels = np.append(keys,["label","color","name","uid"])
        else :
            distribution = get_data_distribution_muscle_for_plot(name,df_inter_motor)
            max_muscle = muscle_keys[np.argmax(distribution)]
            color = muscle_to_color.get(max_muscle)
            color = 'C7' if color is None else color
            labels = np.append(muscle_keys,["label","color",'muscle','name','uid'])
        _ ,_ ,label ,_ ,cluster_color ,_, uids = get_label_intrinsic(name,dataset_brain_inter,data_cluster,brain_to_vnc,vnc)
        cluster_color = cluster_color if motor else color
        if label != None :
            distribution = np.append(distribution,[label, cluster_color,name,uids[0]]) if motor else np.append(distribution,[label, cluster_color,max_muscle,name,uids[0]])
            data_dic=dict(zip(labels,distribution),index = [name])
            data.append(pandas.DataFrame(data_dic))

    data_merged = pandas.concat(data)
    data_no_index = data_merged[keys].values if motor else data_merged[muscle_keys].values
    return data_merged, data_no_index


def inter_clusters_syn_count(vnc,cluster_list,cluster_size_cutoff):
    subnetwork = vnc.subgraph_from_paths(
        source = cluster_list,
        target = cluster_list,
        n_hops = 1,
        keep_edges = 'direct',
    )
    vnc_matrix = subnetwork.get_cmatrix(type_='norm')
    adjacency_mat = subnetwork.get_adjacency_matrix("norm")
    mat = adjacency_mat + adjacency_mat.T
    dense_ = abs(mat*4).todense()
    dense_ = np.clip(dense_,0,1)
    mat = sc.sparse.csr_matrix(dense_)
    mat.setdiag(1.0)

    new_matrix = copy.deepcopy(vnc_matrix)
    new_matrix.matrix = np.abs(mat)
    dense_ = new_matrix.matrix.todense()
    new_matrix.matrix = sc.sparse.csr_matrix(dense_)
    all_clusters = new_matrix.markov_clustering() #clustering
    clusters = [c for c in all_clusters if len(c) >= cluster_size_cutoff]
    uid_clusters = [new_matrix.get_uids(sub_indices=cluster, axis="row") for cluster in clusters]

    return new_matrix,uid_clusters,clusters

def plot_cluster_distribution_list_neurons(uid_list,df_intrinsic, data_cluster, brain_to_vnc_dataset_new):
    labels = ["Anterior grooming", "Take-off/Landing", "Walking", " 4 Flight", "5", "6", "7", "8", "9",
              "Steering", "11 Flight", "12"]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'blueviolet', 'C8', 'C9', 'orangered', 'indigo']
    cluster_list = [0 for i in range(len(labels))]
    fig, ax = plt.subplots()
    for uid in uid_list:
        data_name = df_intrinsic.loc[(df_intrinsic['target_uid'] == uid)]
        cluster_list_2 = plot_cluster_distribution_new(data_name, data_cluster, brain_to_vnc_dataset_new, old_dataset=False,get_distribution=True)
        cluster_list = [cluster_list[i] + cluster_list_2[i] for i in range(len(cluster_list_2))]
        print(cluster_list_2)
    mask = np.where(np.array(cluster_list) > 0, True, False)
    cluster_list_filtered = np.array(cluster_list)[mask]
    labels_filtered = np.array(labels)[mask]
    colors_filtered = np.array(colors)[mask]
    plt.pie(cluster_list_filtered, autopct='%1.1f%%', labels=labels_filtered, colors=colors_filtered)

def get_data_distribution_muscle_for_plot(name : str, df_inter_motor):
    muscle_keys = list(muscle_to_color.keys())
    dict_muscle = dict.fromkeys(muscle_keys,0)
    muscle_data = df_inter_motor.loc[(df_inter_motor['source_name'] == name)]
    muscle_n_list = list(muscle_data['target_muscle'])
    for mn in muscle_n_list:
        if mn not in muscle_keys:
            dict_muscle['Other'] += 1
        else :
            dict_muscle[mn] +=1
    return list(dict_muscle.values())

def plot_muscle_distribution_list_neurons(uid_list,df_intrinsic, df_inter_motor):
    muscle_keys = df_inter_motor["target_muscle"].unique()
    labels = list(muscle_to_color.keys())
    colors = list(muscle_to_color.values())
    cluster_list = [0 for i in range(len(muscle_keys))]
    fig, ax = plt.subplots()
    for uid in uid_list:
        data_name = df_intrinsic.loc[(df_intrinsic['target_uid'] == uid)]['target_name'].iloc[0]
        cluster_list_2 = get_data_distribution_muscle_for_plot(data_name, df_inter_motor)
        cluster_list = [cluster_list[i] + cluster_list_2[i] for i in range(len(cluster_list_2))]
    mask = np.where(np.array(cluster_list) > 0, True, False)
    cluster_list_filtered = np.array(cluster_list)[mask]
    labels_filtered = np.array(labels)[mask]
    print(colors)
    colors_filtered = np.array(colors)[mask]
    plt.pie(cluster_list_filtered, autopct='%1.1f%%', labels=labels_filtered, colors=colors_filtered)

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


Unclear = [
    "Tergopleural/Pleural promotor",
    "Pleural remotor/abductor",
    "Sternal anterior rotator",
    "Sternal posterior rotator",
    "Sternal adductor",
    "Fe reductor",
]
Long_tendon = [
    "ltm",
    "ltm2-femur",
    "ltm1-tibia",
]
Extensors = [
    "Tergotr.",
    "Sternotrochanter",
    "Tr extensor",
    "Ti extensor",
    "Ta depressor",
]
Flexors = [
    "Tr flexor",
    "Acc. tr flexor",
    "Ti flexor",
    "Acc. ti flexor",
    "Ta levator",
]
muscle_to_color = {'Tergopleural/Pleural promotor' : 'g', 'Pleural remotor/abductor' : 'forestgreen', 'Sternal anterior rotator' : 'limegreen',
                   'Sternal posterior rotator' : 'darkgreen', 'Sternal adductor' : 'lime', 'Fe reductor' : 'seagreen',
                   'ltm' : 'navy', 'ltm2-femur' : 'blue', 'ltm1-tibia' : 'skyblue',
                   'Tergotr.' : 'darkred', 'Sternotrochanter' : 'red', 'Tr extensor' : 'salmon', 'Ti extensor' : 'orangered', 'Ta depressor' : 'firebrick',
                   'Tr flexor' : 'violet', 'Acc. tr flexor' : 'purple', 'Ti flexor' : 'darkviolet', 'Acc. ti flexor' : 'magenta', 'Ta levator' : 'deeppink',
                   'Other' : 'tab:gray'
                   }
