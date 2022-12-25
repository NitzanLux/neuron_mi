import os

import matplotlib.pyplot as plt

from utils.evaluations_utils import *
import numpy as np
import matplotlib
import seaborn as sn
#%%{tag:plot name}
# models = {'davids_ergodic_train':'NMDA','reduction_ergodic_train':'reduction','train_AMPA_gmax1':'AMPA gmax 0.0004','train_AMPA_gmax2':'AMPA gmax 0.0008','train_AMPA_gmax3':'AMPA gmax 0.0012','train_AMPA_gmax4':'AMPA gmax 0.0016'}
# name_order = ['NMDA','reduction','AMPA gmax 0.0004','AMPA gmax 0.0008','AMPA gmax 0.0012','AMPA gmax 0.0016']
# file_dest = "small_eval_fnum30000_seed_1623324578916768431.pkl"
models={'Rat_L5b_PC_2_Hay_0-6_DSEN':'L5PC 0.6','Rat_L5b_PC_2_Hay_0-4_DSEN':'L5PC 0.4','Rat_L5b_PC_2_Hay_0-8_DSEN':'L5PC 0.8','Rat_L5b_PC_2_Hay_1_DSEN':'L5PC 1'}
name_order=['L5PC 0.4','L5PC 0.6','L5PC 0.8','L5PC 1']
# with open(os.path.join('entropy_data',file_dest),'rb') as f:
#     d_dict = pickle.load(f)
#%% print(d_dict.keys())
d = ModelsSEData(tags=list(models.keys()),is_folder=True)
# d = ModelsSEData(data_dict=d_dict)
# d.sample_from_set(d_ratio)
#%%
df, m_names = d.get_as_dataframe()
models = {k:v for k,v in models.items()}
# for i in models.keys():
    # df[df['model']==i]['model'] = models[i]
# models = {v:v for v in models.values()}
# temp = {k:v for v,k in models.items()}
# name_order = [temp[i] if i not in models else i for i in name_order]
df.replace(inplace=True,to_replace=models)

#%%


# %% print temporal mean and error
relevant_cols_msx=[]
relevant_cols_ci=[]
representative_msx=set()

for i in df.columns:
    if 'MSx'in i:
        relevant_cols_msx.append(i)
        representative_msx.add(i[:-len('_v_CI')])

    elif 'CI' in i:
        relevant_cols_ci.append(i)
representative_msx = {k:i for i,k in enumerate(representative_msx)}
df = df.sort_values(['key'])
datas = {}
ci_data={}
threshold=None
print(relevant_cols_msx)

for m in tqdm(name_order):
    datas[m]=[]
    ci_data[m]=[]
    for c in relevant_cols_msx:
        print(df[df['model'] == m])
        datas[m].append(np.vstack(df[df['model'] == m][c].tolist()))
    for c in relevant_cols_ci:
        ci_data[m].append(np.vstack(df[df['model'] == m][c].tolist()))

# ci_data = np.hstack(ci_data)
# indexes=np.arange(ci_data[0].shape[0])
sorted(relevant_cols_msx,key = lambda x: representative_msx[x[:-len('_v_CI')]]+int('_v_' in x))
for i,c in enumerate(relevant_cols_msx):
    if i%2==0:
        fig, ax = plt.subplots(1,2)
    for m in tqdm(name_order):
        if threshold is not None:
            mean = np.mean(datas[m][i], axis=0)
            std =  np.std(datas[m][i], axis=0)
        else:
            mean = np.mean(datas[m][i], axis=0)
            std = np.std(datas[m][i], axis=0)
        ax[i%2].plot(np.arange(datas[m][i].shape[1]), mean, label=f'{m}')
        ax[i%2].fill_between(np.arange(datas[m][i].shape[1]), mean - std, mean + std, alpha=0.3)
    ax[i%2].legend()
    # if threshold is not None:
    # ax.set_title(
    # f'Average SE Across Different Time Scales (n = {len(datas[0]) * len(datas):,}) \nCi value {"greater" if direction > 0 else "lower"} than {threshold_value:0.4}')
    # else:
    #     ax.set_title(f'Average SE Across Different Time Scales (n = {len(datas[0]) * len(datas):,})')
    ax[i%2].set_xlabel('Time Scales')
    ax[i%2].set_ylabel('SE value')
    ax[i%2].set_title(c.lower().replace('_',' ').capitalize())
    if i%2==1:
        save_large_plot(fig,f'temporal_msx_{c[:-len("_v_CI")]}_{len(d)}.png',tags=d.data_tags)

    fig
#%%

inf_nan_columns = []
for i in df.columns:
    dfinf = df[i]
    if 'ENTROPY' not in i:
        continue
    if dfinf.dtype == object:
        dfinf = np.array(list(dfinf))

    if np.any(np.isinf(dfinf)):
        inf_nan_columns.append(i)
print(inf_nan_columns)
df.drop(inf_nan_columns, axis=1, inplace=True)
#%%
box_plot_data = {}
colors_index={}
c_obj=set()
for i, m in enumerate(name_order):
    colors_index[m]=f'C{i}'
    for c in df.columns:
        if 'CI'in c:
            c_obj.add(c[:-len('_s_CI')])
for i, m in enumerate(name_order):
    colors_index[m]=f'C{i}'
    for c in df.columns:
        if 'CI'in c:
            if c not in box_plot_data:
                box_plot_data[c]=[]
            box_plot_data[c].append(df[df['model'] == m][c].tolist())
            box_plot_data[c][-1] = np.array(box_plot_data[c][-1])
for c in c_obj:
    fig, ax = plt.subplots()
    bpa=[]
    keylen = len(box_plot_data)
    c_arr=[]
    i=0
    for k in box_plot_data.keys():
        if not c in k:
            continue
        c_arr.append(k)
        for j,m in enumerate(name_order):
            print([j+(i*(len(name_order)+1))])
            bp = ax.boxplot(box_plot_data[k][j],positions=[j+(i*(len(name_order)+1))],patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(colors_index[m])
            bpa.append(bp)
        i+=1

    ax.set_ylabel('Complexity index')
    # ax.set_xticks((np.arange(len(name_order)+1)*len(box_plot_data)))
    a = ax.set_xticks(np.arange(0,(len(name_order)+1)*len(c_arr),len(name_order)+1)+len(name_order)/2-0.5,c_arr)

    # vertical alignment of xtick labels
    # va = [ 0, -.05, 0, -.05, -.05, -.05 ]
    # for t, y in zip( ax.get_xticklabels( ), va ):
    #     t.set_y( y )
    for k,v in colors_index.items():
        ax.plot([],[],color=v,label=k)
    # ax.set_yscale('log')

    # ax.tick_params( axis='x', which='minor', direction='out', length=30 )
    # ax.tick_params( axis='x', which='major', bottom='off', top='off' )
    # ax.set_xticks(np.arange(len(name_order)) + 1, names_for_plots)
    # ax.set_title(f'Sample Entropy Complexity Index (n = {len(box_plot_data[0]) * len(box_plot_data):,})')
    plt.subplots_adjust(bottom=0.4)
    plt.legend()

    fig
    save_large_plot(fig, f"boxplot{c}_{len(d)}.png", tags=d.data_tags)
#%%
box_plot_data = {}
colors_index={}
c_obj=set()
for i, m in enumerate(name_order):
    colors_index[m]=f'C{i}'
    for c in df.columns:
        if 'CI'in c:
            c_obj.add(c[:-len('_s_CI')])
            if c not in box_plot_data:
                box_plot_data[c]=[]
            box_plot_data[c].append(df[df['model'] == name_order[i]][c].tolist())
            box_plot_data[c][-1] = np.array(box_plot_data[c][-1])

# create data
# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents

for i in tqdm(c_obj):
    fig, ax = plt.subplots()
    nbins = 300
    for j,m in enumerate(name_order):
        x,y= box_plot_data[i+'_s_CI'][j],box_plot_data[i+'_v_CI'][j]
        c = ax.scatter(x,y,alpha=0.3,s=0.01)
        ax.plot([],[],color=c.get_facecolor(),label=f'{m}',alpha=1)
    ax.set_ylabel('v')
    ax.set_xlabel('s')
    fig.legend(loc=1)
    ax.set_title(f'v and s scatter plot ci{i}')
    save_large_plot(fig,f'scttervs_{i}_{len(d)}.png',tags=d.data_tags)
    fig
#%%
for c in tqdm(df.columns):
    if not ('ENTROPY' in c and 'CI' in c):
        continue
    df = df.sort_values(['key'])
    datas = []
    ci_data = []
    for i in name_order:
        # datas.append(np.vstack(df[df['model'] == i][c].tolist()))
        ci_data.append(np.vstack(df[df['model'] == i][c].tolist()))
    dist = []
    bins = 200
    # all_data= np.vstack(datas)
    fig, ax = plt.subplots()
    for j, n in enumerate(name_order):
        frequency, bins = np.histogram(ci_data[j], bins=bins)
        frequency = frequency / np.sum(frequency)
        ax.stairs(frequency, bins, fill=True, label=n, alpha=0.4)
    fig.legend(loc=1, borderaxespad=3)
    ax.set_title(c.lower().replace('_', ' ').capitalize())
    ax.set_ylabel('P(ci)')
    ax.set_xlabel('ci value')
    save_large_plot(fig,f'dist_1d_{c}_{len(d)}.png',tags=d.data_tags)
    fig
#%%
box_plot_data = {}
colors_index={}
c_obj=set()
for i, m in enumerate(name_order):
    colors_index[m]=f'C{i}'
    for c in df.columns:
        if 'CI'in c:
            c_obj.add(c[:-len('_s_CI')])
for i, m in enumerate(name_order):
    colors_index[m]=f'C{i}'
    for c in df.columns:
        if 'CI'in c:
            if c not in box_plot_data:
                box_plot_data[c]=[]
            box_plot_data[c].append(df[df['model'] == m][c].tolist())
            box_plot_data[c][-1] = np.array(box_plot_data[c][-1])
for c in tqdm(c_obj):
    # for i,m in enumerate(name_order):
    res = sn.kdeplot(data=df,y = f'{c}_v_CI',x =f'{c}_s_CI', shade=False,hue='model',alpha=0.6)#thresh
    plt.ylabel('v')
    plt.xlabel('s')
    plt.title(c.lower().replace('_',' ').capitalize())
    save_large_plot(plt,f'kde{c}_{len(d)}.png',tags=d.data_tags)
    plt.show()