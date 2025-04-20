from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

from sklearn.metrics.pairwise import cosine_similarity

x = [1,2,3, 4, 0.09728274,-0.03882077]

a=0

for i in range(len(x)):
    a+=x[i]

print('aaaa',a)





dis = cosine_similarity(X=x, Y=y)

print(dis)



"""
    产生数据
"""
def create_data(centers,num=100,std=0.7):
    X,labels_true = make_blobs(n_samples=num,centers=centers, cluster_std=std)
    return X,labels_true

"""
    数据作图
"""
def plot_data(*data):
    X,labels_true=data
    labels=np.unique(labels_true)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors='rgbycm'
    for i,label in enumerate(labels):
        position=labels_true==label
        ax.scatter(X[position,0],X[position,1],label="cluster %d"%label),
        color=colors[i%len(colors)]

    ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[1]")
    ax.set_title("data")
    plt.show()




"""
    测试函数
"""
def test_AgglomerativeClustering(*data):
    X,labels_true=data
    clst=cluster.AgglomerativeClustering()
    predicted_labels=clst.fit_predict(X)
    print("ARI:%s"% adjusted_rand_score(labels_true, predicted_labels))

"""
    考察簇的数量对于聚类效果的影响
"""



def test_AgglomerativeClustering_nclusters(*data):
    X,labels_true=data
    nums=range(1,30)
    ARIS=[]
    for num in nums:
        clst=cluster.AgglomerativeClustering(affinity="cosine",n_clusters=num,linkage="complete")
        predicted_lables=clst.fit_predict(-X)
        ARIS.append(adjusted_rand_score(labels_true, predicted_lables))

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(nums,ARIS,marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    fig.suptitle("AgglomerativeClustering")
    plt.show()

"""
    考察链接方式对聚类结果的影响
"""
def test_agglomerativeClustering_linkage(*data):
    X,labels_true=data
    nums=range(1,50)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    linkages=['ward','complete','average']
    markers="+o*"
    for i,linkage in enumerate(linkages):
        ARIs=[]
        for num in nums:
            clst=cluster.AgglomerativeClustering(n_clusters=num,linkage=linkage)
            predicted_labels=clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        ax.plot(nums,ARIs,marker=markers[i],label="linkage:%s"%linkage)

    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    fig.suptitle("AgglomerativeClustering")
    plt.show()

centers=[[1,1],[2,2],[1,2],[10,20]]
X,labels_true=create_data(centers, 1000, 0.5)
print(X)
print(X.shape)




test_AgglomerativeClustering(X,labels_true)
plot_data(X,labels_true)
test_AgglomerativeClustering_nclusters(X,labels_true)
#test_agglomerativeClustering_linkage(X,labels_true)


cluster_indices_new = []

def cluster_clients(S):
    clustering =cluster.AgglomerativeClustering(affinity="cosine", linkage="complete").fit(-S)
    c1 = np.argwhere(clustering.labels_ == 0).flatten()
    c2 = np.argwhere(clustering.labels_ == 1).flatten()
    return c1, c2
#
# c1, c2 = cluster_clients(X)
# print(c2)
# print('cs',c1)

#
# cluster_indices_new += [c1, c2]
#
#
# from helper import ExperimentLogger, display_train_stats
# from fl_devices import Server, Client
# server=Server
#
# from sklearn.cluster import AgglomerativeClustering
# #
# # cluster_indices_new = []
#
#
#
# c1, c2 = cluster_clients(X)
#
# cluster_indices_new += [c1, c2]
#
#
#
# np.random.seed(42)
#
# cfl_stats = ExperimentLogger()
#
# clients= np.array(X)
#
# COMMUNICATION_ROUNDS=8
#
# cluster_indices = [np.arange(len(clients)).astype("int")]
# client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
#
# for c_round in range(1, COMMUNICATION_ROUNDS + 1):
#     # if c_round == 1:
#     #     for client in clients:
#     #         client.synchronize_with_server(server)
#     # participating_clients = server.select_clients(clients, frac=1.0)
#
#     # for client in participating_clients:
#     #     train_stats = client.compute_weight_update(epochs=1)
#     #     client.reset()
#
#     similarities = server.compute_pairwise_similarities(None,clients)
#
#     cluster_indices_new = []
#     for idc in cluster_indices:
#         max_norm = 0.55  # server.compute_max_update_norm(self,[clients[i] for i in idc])
#         mean_norm = 0.88  # server.compute_mean_update_norm(self,[clients[i] for i in idc])
#
#         if c_round > 5:
#
#             # server.cache_model(idc, clients[idc[0]], acc_clients)
#
#             c1, c2 = server.cluster_clients(similarities)
#             cluster_indices_new += [c1, c2]
#
#             cfl_stats.log({"split": c_round})
#
#         else:
#             cluster_indices_new += [idc]
#
#     cluster_indices = cluster_indices_new
#     # self.client_clusters = [[clients[i] for i in range(idcs)] for idcs in range(cluster_indices)]
#     #
#     # server.aggregate_clusterwise(self,self.client_clusters)
#
#     # clients= np.array(clients)
#     # clients.astype(float)
#     # clients = torch.Tensor(clients)
#
#     # 为什么不能evaluate
#
#     acc_clients = [client.evaluate() for client in clients]
#
#     cfl_stats.log({"acc_clients": acc_clients, "rounds": c_round, "mean_norm": mean_norm, "max_norm": max_norm,
#                    "clusters": cluster_indices})  #
#
#     EPS_1 = 0.4
#     EPS_2 = 1.6
#
#     display_train_stats(cfl_stats, EPS_1, EPS_2, COMMUNICATION_ROUNDS)

