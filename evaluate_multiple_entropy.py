from utils.evaluations_utils import *

# %%
models = {'davids_ergodic_train_fnum_580_snum74240':'NMDA','reduction_ergodic_train_fnum_580_snum74187':'reduction'}

name_order = ['NMDA','reduction']
names_for_plots= name_order
d = ModelsSEData(models.keys())
df, m_names = d.get_as_dataframe()
models = {k[:k.find("_fnum")]:v for k,v in models.items()}
temp = {k:v for v,k in models.items()}
name_order = [temp[i] if i not in models else i for i in name_order]
#%% print nans ci
inf_dist_columns = []
for i in df.columns:
    dfinf = df[i]
    if 'ENTROPY' not in i:
        inf_dist_columns.append(0)
        continue
    if dfinf.dtype == object:
        dfinf = np.array(list(dfinf))
        inf_dist_columns.append(np.any(np.isnan(dfinf),axis=1).sum()/dfinf.shape[0])
        continue
    inf_dist_columns.append(np.isnan(dfinf).sum()/dfinf.shape[0])
fig,ax=plt.subplots()
ax.bar(range(len(inf_dist_columns)),inf_dist_columns)
ax.set_xticks(np.arange(len(inf_dist_columns)),df.columns,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.show()

# %% print infs in array
inf_dist_columns = []
for i in df.columns:
    dfinf = df[i]
    if 'ENTROPY' not in i:
        inf_dist_columns.append(0)
        continue
    if dfinf.dtype == object:
        dfinf = np.array(list(dfinf))
        inf_dist_columns.append(np.any(np.isinf(dfinf),axis=1).sum()/dfinf.shape[0])
        continue
    inf_dist_columns.append(np.isinf(dfinf).sum()/dfinf.shape[0])
fig,ax=plt.subplots()
ax.bar(range(len(inf_dist_columns)),inf_dist_columns)
ax.set_xticks(np.arange(len(inf_dist_columns)),df.columns,rotation=90)
plt.subplots_adjust(bottom=0.4)
plt.show()
# %% remove infs and update
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
#%%
df.drop(inf_nan_columns,axis=1,inplace=True)
# %% box plot complexity
fig, ax = plt.subplots()
box_plot_data = {}
colors_index={}

for i, m in enumerate(name_order):
    colors_index[m]=f'C{i}'
    for c in df.columns:
        if 'CI'in c:
            if c not in box_plot_data:
                box_plot_data[c]=[]
            box_plot_data[c].append(df[df['model'] == name_order[i]][c].tolist())
            box_plot_data[c][-1] = np.array(box_plot_data[c][-1])

bpa=[]
keylen = len(box_plot_data)
c_arr=[]
for i,k in enumerate(box_plot_data.keys()):
    c_arr.append(k)
    for j,m in enumerate(name_order):
        bp = ax.boxplot(box_plot_data[k],positions=[j+(i*(len(name_order)+1))], showfliers=False)
        bp['color'] = colors_index[m]
        bpa.append(bp)
ax.set_ylabel('Complexity index')
# ax.set_xticks((np.arange(len(name_order)+1)*len(box_plot_data)))
print(np.arange(0,(len(name_order)+1)*len(box_plot_data),len(m_names)))
ax.set_xticks(np.arange(0,(len(name_order)+1)*len(box_plot_data),len(name_order)+1),c_arr,rotation=45)
# vertical alignment of xtick labels
va = [ 0, -.05, 0, -.05, -.05, -.05 ]
for t, y in zip( ax.get_xticklabels( ), va ):
    t.set_y( y )
plt.subplots_adjust(bottom=0.4)
# ax.tick_params( axis='x', which='minor', direction='out', length=30 )
# ax.tick_params( axis='x', which='major', bottom='off', top='off' )
# ax.set_xticks(np.arange(len(name_order)) + 1, names_for_plots)
# ax.set_title(f'Sample Entropy Complexity Index (n = {len(box_plot_data[0]) * len(box_plot_data):,})')
# save_large_plot(fig, "boxplot.png", name_order)
fig.show()

# %% normelized box plot todo implement
# fig, ax = plt.subplots()
# normelized_box_plot_data = []
# for i, m in enumerate(name_order):
#     normelized_box_plot_data.append(df[df['model'] == name_order[i]]['Ci'].tolist())
#     normelized_box_plot_data[-1] = np.array(normelized_box_plot_data[-1])
# for i in range(len(normelized_box_plot_data[0])):
#     cur_mean = sum([d[i] for d in normelized_box_plot_data]) / len(normelized_box_plot_data)
#     cur_std = np.std([d[i] for d in normelized_box_plot_data])
#     for j in range(len(normelized_box_plot_data)):
#         normelized_box_plot_data[j][i] -= cur_mean
#         normelized_box_plot_data[j][i] / cur_std
#
#     # normelized_box_plot_data[-1]=normelized_box_plot_data[-1][normelized_box_plot_data[-1]>100]
# p01 = ttest_ind(normelized_box_plot_data[0], normelized_box_plot_data[1], equal_var=False).pvalue
# p12 = ttest_ind(normelized_box_plot_data[2], normelized_box_plot_data[1], equal_var=False).pvalue
# p02 = ttest_ind(normelized_box_plot_data[0], normelized_box_plot_data[2], equal_var=False).pvalue
# print(p01, p12, p02)
# ax.boxplot(normelized_box_plot_data, showfliers=False)
# ax.set_ylabel('sample entropy complexity index')
# # ax.set_xticks(np.arange(len(name_order)) + 1, names_for_plots)
# ax.set_title(
#     f'Sample Entropy Complexity Index \nZ-Score per trial (n = {len(normelized_box_plot_data[0]) * len(normelized_box_plot_data):,})')
# save_large_plot(fig, "boxplot_normelized.png", name_order)
# fig.show()
#%% scatter plot of the cis
fig, ax = plt.subplots()
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
nbins = 300
for i in tqdm(c_obj):
    for j,m in tqdm(enumerate(name_order)):
        x,y= box_plot_data[i+'_s_CI'][j],box_plot_data[i+'_v_CI'][j]
        c = ax.scatter(x,y,alpha=0.3,s=0.01)
        ax.plot([],[],color=c.get_facecolor(),label=f'{i}_{m}',alpha=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.legend()
    plt.show()

# %% spike_count
plt.close()
fig, ax = plt.subplots()
datas = []
for i in name_order:
    datas.append(np.array(df[df['model'] == i]['spikes'].tolist()).sum(axis=1))
    # a=np.mean(np.array(datas[-1]),axis=1)
    # print(i, np.mean(np.array(datas[-1]),axis=1))
names_for_plots_labels = [(mn + f" ($\mu$ = {np.round(np.mean(np.array(datas[i])), 2)})") for i, mn in
                          enumerate(names_for_plots)]
bins = np.arange(30) - 0.5


ax.hist(datas, bins=bins,rwidth=0.75, label=names_for_plots_labels, alpha=1, align='mid')
ax.set_title('Spike Count Per Trial')
# ax.set_yscale('log')
ax.set_xlabel('Number of Spikes in trail')
ax.set_ylabel('Number of Occurrences')
ax.set_xticks(bins[1:(8 * (bins.shape[0] // 8)):2] - 0.5, np.arange(0, (8 * (bins.shape[0] // 8)), 2))
fig.legend(bbox_to_anchor=[0.9, 0.875])
# save_large_plot(fig, 'spike_count.png', name_order)
fig.show()

# %% scatter plot pairwise complaxity plots[3].
df = df.sort_values(['key'])
datas = []
for i in name_order:
    datas.append(df[df['model'] == i]['Ci'].tolist())
for i in range(3):
    first_index = i
    second_index = (i + 1) % 3
    first_index, second_index = min([first_index, second_index]), max([first_index, second_index])
    fig, ax = plt.subplots()
    ax.scatter(datas[first_index], datas[second_index], alpha=0.2, s=0.1)
    ax.set_xlabel(names_for_plots[first_index])
    ax.set_ylabel(names_for_plots[second_index])
    lims = (np.min(np.vstack((datas[first_index], datas[second_index]))),
            np.max(np.vstack((datas[first_index], datas[second_index]))))
    ax.plot(lims, lims, color='red')
    ax.set_title(
        f"{names_for_plots[first_index]} and {names_for_plots[second_index]} trials \nSE complexity index (n = {len(datas[second_index]) * 2:,})")
    save_large_plot(fig, "pairwise_scatter.png", [name_order[first_index], name_order[second_index]])

    fig.show()
# %% 2d histogram pairwise plots[3].
df = df.sort_values(['key'])
datas = []
for i in name_order:
    datas.append(df[df['model'] == i]['Ci'].tolist())

for i in range(3):
    first_index = i
    second_index = (i + 1) % 3
    first_index, second_index = min([first_index, second_index]), max([first_index, second_index])
    fig, ax = plt.subplots()
    lims = (np.min(np.vstack((datas[first_index], datas[second_index]))),
            np.max(np.vstack((datas[first_index], datas[second_index]))))

    H, xedges, yedges = np.histogram2d(datas[first_index], datas[second_index], range=np.array([lims, lims]),
                                       bins=int(lims[1] - lims[0]))
    # replace zeroes with nan
    H[H == 0] = np.nan
    im = ax.imshow(H.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
                   )
    # , norm=colors.LogNorm())
    ax.plot(lims, lims, color='black')

    slope, intercept, r_value, p_value, std_err = linregress(np.array(datas[first_index]), datas[second_index])
    reg_intercep = intercept
    reg_coef = slope
    x = lims
    y = [reg_intercep + lims[0] * reg_coef, reg_intercep + lims[1] * reg_coef]
    ax.plot(x, y, color='red')
    fig.colorbar(im)
    ax.set_xlabel(names_for_plots[first_index])
    ax.set_ylabel(names_for_plots[second_index])
    ax.set_ylim(lims)
    ax.set_xlim(lims)
    print(p_value)
    ax.set_title(
        f"{names_for_plots[first_index]} and {names_for_plots[second_index]} trials SE Complexity Index \nn = {len(datas[second_index]) * 2:,} ,$R^2$ = {np.round(r_value ** 2, 4)}")
    save_large_plot(fig, "pairwise_2dhist.png", [name_order[first_index], name_order[second_index]])

    fig.show()

# %% 2d histogram pairwise plots[3] firing rate.
df = df.sort_values(['key'])
datas = []
for i in name_order:
    datas.append(df[df['model'] == i]['spike_number'].tolist())

for i in range(3):
    first_index = i
    second_index = (i + 1) % 3
    first_index, second_index = min([first_index, second_index]), max([first_index, second_index])
    fig, ax = plt.subplots()
    lims = (np.min(np.vstack((datas[first_index], datas[second_index]))),
            np.max(np.vstack((datas[first_index], datas[second_index]))))

    H, xedges, yedges = np.histogram2d(datas[first_index], datas[second_index], range=np.array([lims, lims]),
                                       bins=int(lims[1] - lims[0]))
    # replace zeroes with nan
    H[H == 0] = np.nan
    im = ax.imshow(H.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
                   )
    # , norm=colors.LogNorm())
    ax.plot(lims, lims, color='black')

    slope, intercept, r_value, p_value, std_err = linregress(np.array(datas[first_index]), datas[second_index])
    reg_intercep = intercept
    reg_coef = slope
    x = lims
    y = [reg_intercep + lims[0] * reg_coef, reg_intercep + lims[1] * reg_coef]
    ax.plot(x, y, color='red')
    fig.colorbar(im)
    ax.set_xlabel(names_for_plots[first_index])
    ax.set_ylabel(names_for_plots[second_index])
    ax.set_ylim(lims)
    ax.set_xlim(lims)
    print(p_value)
    ax.set_title(
        f"{names_for_plots[first_index]} and {names_for_plots[second_index]} trials SE Complexity Index \nn = {len(datas[second_index]) * 2:,} ,$R^2$ = {np.round(r_value ** 2, 4)}")
    save_large_plot(fig, "pairwise_2dhist.png", [name_order[first_index], name_order[second_index]])

    fig.show()

# %% plot difference avarage per file
df = df.sort_values(['key'])
fig, ax = plt.subplots()
datas = []

for i in name_order:
    datas.append(np.vstack(df[df['model'] == i]['SE'].tolist()))
# avarage_diff = []

for i in range(3):
    first_index = i
    second_index = (i + 2) % 3
    first_index, second_index = min([first_index, second_index]), max([first_index, second_index])
    assert first_index != second_index
    diff = datas[first_index] - datas[second_index]
    mean = np.mean(diff, axis=0)
    std = np.std(diff, axis=0)
    ax.plot(np.arange(diff.shape[1]), mean,
            label=names_for_plots[first_index] + " - " + names_for_plots[second_index], )
    ax.fill_between(np.arange(diff.shape[1]), mean - std, mean + std, alpha=0.3)
ax.set_title(f'Differences Across Different SE Scales (n = {len(datas[0]) * len(datas):,})')
ax.set_xlabel('Time Scales')
ax.set_ylabel('Differences')
plt.legend()
plt.tight_layout()

save_large_plot(fig, 'differences_between_the_same_inputs.png', name_order)
plt.show()

# %% print temporal mean and error
relevant_cols_msx=[]
relevant_cols_ci=[]
for i in df.columns:
    if 'MSx'in i:
        relevant_cols_msx.append(i)
    elif 'CI' in i:
        relevant_cols_ci.append(i)
# threshold = None
# direction = -1
# box_plot_data = np.array(box_plot_data)
# threshold_box_plot = np.sort(box_plot_data, axis=1)
# if direction > 1 and threshold is not None:
#     threshold_ration = threshold
# elif threshold is not None:
#     threshold_ration = 1 - threshold
# if threshold is not None:
#     threshold_value = np.min(threshold_box_plot[:, int(threshold_box_plot.shape[1] * (threshold_ration))])

df = df.sort_values(['key'])
datas = {}
ci_data={}
threshold=None
# from scipy.stats import sem

for m in tqdm(name_order):
    datas[m]=[]
    ci_data[m]=[]
    for c in relevant_cols_msx:
        datas[m].append(np.vstack(df[df['model'] == m][c].tolist()))
    for c in relevant_cols_ci:
        ci_data[m].append(np.vstack(df[df['model'] == m][c].tolist()))

# ci_data = np.hstack(ci_data)
# indexes=np.arange(ci_data[0].shape[0])
for i,c in enumerate(relevant_cols_msx):
    fig, ax = plt.subplots()
    for m in tqdm(name_order):
        print(c)
        if threshold is not None:
            mean = np.mean(datas[m][i], axis=0)
            std =  np.std(datas[m][i], axis=0)
        else:
            mean = np.mean(datas[m][i], axis=0)
            std = np.std(datas[m][i], axis=0)
        ax.plot(np.arange(datas[m][i].shape[1]), mean, label=f'{m}_{c}')
        ax.fill_between(np.arange(datas[m][i].shape[1]), mean - std, mean + std, alpha=0.3)
    ax.legend(loc='upper left')
    # if threshold is not None:
    # ax.set_title(
    # f'Average SE Across Different Time Scales (n = {len(datas[0]) * len(datas):,}) \nCi value {"greater" if direction > 0 else "lower"} than {threshold_value:0.4}')
    # else:
    #     ax.set_title(f'Average SE Across Different Time Scales (n = {len(datas[0]) * len(datas):,})')
    ax.set_xlabel('Time Scales')
    ax.set_ylabel('SE value')
    # if threshold is not None:
    #     save_large_plot(fig,
    #                     f'Average_SE_across_different_Time_Scales_th_{direction}{str(threshold).replace(".", "!")}.png',
    #                     name_order)
    # else:
    #     save_large_plot(fig, f'Average_SE_across_different_Time_Scales.png', name_order)
    plt.show()

# %% plot files by order:


df = df.sort_values(['key'])

diff_vec = []
for j in tqdm(name_order):
    a = df[(df['model'] == j)]['Ci'].values
    diff_vec.append(df[(df['model'] == j)]['Ci'].values)
diff_vec = np.array(diff_vec)
temp_diff_vec = diff_vec.copy()
a = np.argsort(np.linalg.norm(diff_vec, axis=0))
fig, ax = plt.subplots()
mat = diff_vec[:, a]
im = ax.matshow(mat)
ax.set_aspect(3000)
ax.set_xticks([])
ax.set_xlabel(f'Sorted Trial index (n = {diff_vec.shape[1]:,})')
ax.set_yticks(range(len(name_order)), names_for_plots)
ax.set_title('Norm Sorted SE Complexity Index')
fig.colorbar(im, location="bottom")

plt.tight_layout()

save_large_plot(fig, 'norm_wise_orderd_matrix.png', name_order)

fig.show()
# %% plot files by order 3d view
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i, m in enumerate(name_order):
    ax.scatter(np.arange(diff_vec.shape[1]), diff_vec[i, a], np.ones((diff_vec.shape[1])) * i,
               label=names_for_plots_dict[m], alpha=0.6,
               s=0.5)
ax.set_xlabel(f'Sorted Trial index (n = {diff_vec.shape[1]:,})')
ax.set_ylabel('SE Complexity Index')
ax.set_xticks([])
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks(range(len(name_order)), names_for_plots)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)
ax.spines.right.set_visible(False)
ax.set_title('Norm Sorted SE Complexity Index')
# fig.legend(loc='lower left')
# plt.tight_layout()
save_large_plot(fig, 'norm_wise_orderd_matrix3d.png', name_order)

fig.show()
# %% correlation matrix
dummies_df = pd.get_dummies(df[['Ci', 'spike_number', 'model']], columns=['model'])
# dummies_df=df[['Ci','spike_number','model']]
corr = dummies_df.corr()
# %%
fig, ax = plt.subplots()
mat = corr.to_numpy()
cols = list(corr.columns)
cols = [names_for_plots_dict[c[len('model_'):]] if (c[len('model_'):] in names_for_plots_dict) else c for c in cols]
mat[np.arange(mat.shape[0]), np.arange(mat.shape[0])] = np.NAN
minmax_val = np.max([np.abs(np.nanmin(mat)), np.nanmax(mat)])
cmap = matplotlib.cm.get_cmap('bwr').copy()
cmap.set_bad('black', 1.)
divnorm = colors.TwoSlopeNorm(vmin=-float(minmax_val), vcenter=0., vmax=float(minmax_val))
im = ax.matshow(mat, cmap=cmap, norm=divnorm)
for i in range(mat.shape[0]):
    for j in range(mat.shape[0]):
        c = mat[j, i]
        if i == j: c = 1

        ax.text(i, j, '%0.4f' % c, va='center', ha='center')
ax.set_xticks(range(len(cols)), cols, rotation=45, fontsize=8)
ax.set_yticks(range(len(cols)), cols, rotation=45, fontsize=8)
ax.set_title('Cross Correlation matrix')
ax.grid(False)
plt.colorbar(im)
plt.tight_layout()
save_large_plot(fig, 'cross_correlation.png', name_order)
plt.show()
# %%
data = []
df = df.sort_values(['key'])

for i in name_order:
    data.append(df[df['model'] == i]['Ci'].to_numpy())
cols = names_for_plots
data = np.vstack(data)
mat = np.corrcoef(data)
fig, ax = plt.subplots()

mat[np.arange(mat.shape[0]), np.arange(mat.shape[0])] = np.NAN
minmax_val = np.max([np.abs(np.nanmin(mat)), np.nanmax(mat)])

cmap = matplotlib.cm.get_cmap('bwr').copy()
cmap.set_bad('black', 1.)
if np.sign(np.nanmin(mat)) == np.sign(np.nanmax(mat)):
    min_val = np.nanmin(mat)
    max_val = np.nanmax(mat)
else:
    min_val = -float(minmax_val)
    max_val = float(minmax_val)
divnorm = colors.TwoSlopeNorm(vmin=min_val, vcenter=(min_val + max_val) / 2, vmax=max_val)
im = ax.matshow(mat, cmap=cmap, norm=divnorm)
for i in range(mat.shape[0]):
    for j in range(mat.shape[0]):
        c = mat[j, i]
        if i == j: c = 1

        ax.text(i, j, '%0.4f' % c, va='center', ha='center')
ax.set_xticks(range(len(cols)), cols, rotation=45, fontsize=8)
ax.set_yticks(range(len(cols)), cols, rotation=45, fontsize=8)
ax.set_title('Cross Correlation matrix')
ax.grid(False)
plt.colorbar(im)
plt.tight_layout()
save_large_plot(fig, 'cross_correlation_between_different_models.png', name_order)
plt.show()

# %% plot distributions.

df = df.sort_values(['key'])
datas = []
ci_data = []
for i in name_order:
    datas.append(np.vstack(df[df['model'] == i]['SE'].tolist()))
    ci_data.append(np.vstack(df[df['model'] == i]['Ci'].tolist()))
dist = []
bins = 100
all_data= np.vstack(datas)
da_np=da.from_array(all_data,chunks=(372*3,all_data.shape[1]))
u, s, vh = da.linalg.svd(da_np)
    # ax[1 + k * 2].stairs(s, np.arange(s.shape[0]+1))
s=np.array(s)
#%%
fig, ax = plt.subplots(3)
for j, n in tqdm(enumerate(name_order)):
    frequency, bins = np.histogram(ci_data[j], bins=bins)
    frequency = frequency / np.sum(frequency)
    ax[0].stairs(frequency, bins, fill=True, label=n, alpha=0.5)

# for k, n in tqdm(enumerate(name_order)):

ax[1].bar(x = np.arange(s.shape[0]),height=s ,fill=True)

    # ax.plot(u, label=np.arange(s.shape[0]))
for i in range(3):
    p = ax[2].plot(vh[i,:],alpha=0.5,label=i)
# fig.legend()
fig.tight_layout()
fig.show()
# %%
from pandas.plotting import andrews_curves

s_i = df['SE'].copy()
ddf = df.iloc[:, df.columns != 'SE'].copy()
s_i = np.vstack(s_i.to_numpy())
for i in range(s_i.shape[1]):
    ddf.insert(len(ddf.columns), 'SE%d' % i, s_i[:, i])
ddf['file'] = pd.factorize(ddf['file'])[0]
ddf.drop(columns=['key'])
# %%
andrews_curves(ddf, 'model')
