from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import Normalize



def pca_plotter(data, target):
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(data)

    # cmap = cm.rainbow
    target_set = list(set(target))
    # if len(target_set)>1:
    #     norm = Normalize(vmin=0, vmax=len(target_set)-1)
    #     print(norm(target_set.index(target[0])))
    #     target_colors = [cmap(norm(target_set.index(t))) for t in target]
    # else:
    #     clr = cmap(0.5)
    #     target_colors = [clr for t in target]
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(Xt[:,0], Xt[:,1], c=[target_set.index(t) for t in target], cmap=cm.rainbow)
    print(len(scatter.legend_elements()[0]))
    print(target_set)
    #ax.legend(handles=scatter.legend_elements()[0], labels=target_set)
    ax.legend(handles=scatter.legend_elements()[0], labels = target_set, frameon=True)
    return fig