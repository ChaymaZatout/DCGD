import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:

    def viz_cut(self, cut):
        pass

    @staticmethod
    def viz_cuts(cuts, filename="", show=False):
        intermediate_list = [el for els in cuts for el in els]
        # show simple scatter:
        df = pd.DataFrame(intermediate_list, columns=['x', 'y', 'y_real'])
        plt.clf()
        sns.set(style='darkgrid')
        ax = sns.scatterplot(x="x", y="y_real", data=df, palette="Set2", linewidth=0)

        if len(filename) > 0:
            # save file:
            plt.savefig(filename)
        if show:
            plt.show()

    @staticmethod
    def viz_subcuts(subcuts, filename="", show=False):
        intermediate_list = [el + [i] for i in range(len(subcuts)) for el in subcuts[i]]
        df = pd.DataFrame(intermediate_list, columns=['x', 'y', 'y_real', 'group'])
        # show:
        plt.clf()
        ax = sns.scatterplot(x="x", y="y_real", data=df, hue="group", palette="Set2", linewidth=0)

        if len(filename) > 0:
            # save file:
            plt.savefig(filename)

        if show:
            plt.show()

    @staticmethod
    def viz_subcuts_range(subcuts, interval, filename="", show=False):
        intermediate_list = [el + [i] for i in range(len(subcuts)) for el in subcuts[i]]
        df = pd.DataFrame(intermediate_list, columns=['x', 'y', 'y_real', 'group'])
        # show:
        plt.clf()
        fig, ax = plt.subplots(1, 2)
        sns.set(style='darkgrid')
        sns.scatterplot(x="x", y="y_real", data=df, hue="group", palette="Set2", linewidth=0, ax=ax[0])
        sns.scatterplot(x="x", y="y_real", data=df.loc[df['group'].isin(interval)], hue="group", palette="Set2",
                        linewidth=0, ax=ax[1])

        if len(filename) > 0:
            # save file:
            plt.savefig(filename)

        if show:
            plt.show()

    @staticmethod
    def viz_labels(labels, filename="", show=False):
        intermediate_list = []
        for i in range(len(labels)):
            for p in labels[i][1]:
                intermediate_list.append(p + [labels[i][0]])

        df_noise = pd.DataFrame(intermediate_list, columns=['x', 'y', 'y_real', 'label'])
        plt.clf()
        sns.set(style='darkgrid')
        sns.scatterplot(x="x", y="y_real", data=df_noise, hue="label", palette="Set2", linewidth=0)

        if len(filename) > 0:
            # save file:
            plt.savefig(filename)

        if show:
            plt.show()

    @staticmethod
    def viz_filtred_labels(labels, noise, filename="", show=False):
        intermediate_list = []
        for i in range(len(labels)):
            for p in labels[i][1]:
                intermediate_list.append(p + [labels[i][0]])
        for i in range(len(noise)):
            for p in noise[i]:
                intermediate_list.append(p + ["noise"])
        df_noise = pd.DataFrame(intermediate_list, columns=['x', 'y', 'y_real', 'label'])
        plt.clf()
        sns.set(style='darkgrid')
        sns.scatterplot(x="x", y="y_real", data=df_noise, hue="label", palette="Set2", linewidth=0)

        if len(filename) > 0:
            # save file:
            plt.savefig(filename)

        if show:
            plt.show()

    @staticmethod
    def viz_all_labels(alllabels, interval, filename="", show=False):
        intermediate_list = []

        for labels in alllabels:
            for i in range(len(labels)):
                for p in labels[i][1]:
                    intermediate_list.append([i + min(interval)] + p + [labels[i][0]])

        df = pd.DataFrame(intermediate_list, columns=['depth', 'x', 'y', 'y_real', 'label'])
        plt.clf()
        sns.set(style='darkgrid')
        sns.scatterplot(x="x", y="y_real", data=df, hue="label", palette="Set2", linewidth=0)

        if len(filename) > 0:
            # save file:
            plt.savefig(filename)

        if show:
            plt.show()

    @staticmethod
    def viz_all_filtered_labels(alllabels, allnoise, interval, filename="", show=False):
        intermediate_list = []
        for labels in alllabels:
            for i in range(len(labels)):
                for p in labels[i][1]:
                    intermediate_list.append([i + min(interval)] + p + [labels[i][0]])
        for i in range(len(allnoise)):
            noise = allnoise[i]
            for n in noise:
                for p in n:
                    intermediate_list.append([i] + p + ["noise"])
        df = pd.DataFrame(intermediate_list, columns=['depth', 'x', 'y', 'y_real', 'label'])
        plt.clf()
        sns.set(style='darkgrid')
        sns.scatterplot(x="x", y="y_real", data=df, hue="label", palette="Set2", linewidth=0)

        if len(filename) > 0:
            # save file:
            plt.savefig(filename)

        if show:
            plt.show()

    @staticmethod
    def viz_all_filtered_labels_(alllabels, allnoise, filename="", show=False):
        intermediate_list = []
        for i in range(len(alllabels)):
            labels = alllabels[i]
            for i in range(len(labels)):
                for p in labels[i][1]:
                    intermediate_list.append([i] + p + [labels[i][0]])
        for i in range(len(allnoise)):
            noise = allnoise[i]
            for n in noise:
                for p in n:
                    intermediate_list.append([i] + p + ["noise"])
        df = pd.DataFrame(intermediate_list, columns=['depth', 'x', 'y', 'y_real', 'label'])
        plt.clf()
        sns.set(style='darkgrid')
        sns.scatterplot(x="x", y="y_real", data=df, hue="label", palette="Set2", linewidth=0)

        if len(filename) > 0:
            # save file:
            plt.savefig(filename)

        if show:
            plt.show()

    @staticmethod
    def viz_on_depth(pretty, alllabels, allnoise=None):
        if allnoise is None:
            allnoise = []

        for labels in alllabels:
            for a in labels:
                for p in a[1]:
                    if a[0] == "cc":
                        pretty[p[1], p[0]] = [255, 0, 255]
                    else:
                        # pretty[p[1], p[0]] = [0, 0, 255]
                        pass
        for noise in allnoise:
            for n in noise:
                for p in n:
                    # pretty[p[1], p[0]] = [255, 0, 0]
                    pass


    @staticmethod
    def viz_on_depth_downsampling(pretty, depth, interval, h_error, step, alllabels, allnoise=None,
                                  cy_d=0, fy_d=1):
        if allnoise is None:
            allnoise = []

        (mini, maxi) = interval
        (height, width) = depth.shape[:2]
        for x in range(width):
            for y in range(height):
                d = depth[y, x]
                if mini <= d <= maxi:
                    ind = int((d - mini) / step)
                    labels = alllabels[ind]
                    for a in labels:
                        for p in a[1]:
                            if x == p[0]:
                                yreal = - ((y - cy_d) * d / fy_d)  # compute the real y
                                if abs(yreal - p[2]) <= h_error*2:
                                    if a[0] == "cc":
                                        pretty[y, x] = [0, 255, 0]
                                    else:
                                        pass
                                        # pretty[y, x] = [0, 0, 255]
                                else:
                                    pass
                                    # pretty[y, x] = [0, 255, 255]
