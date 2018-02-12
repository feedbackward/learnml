
# coding: utf-8

# # エンコーダ用の階層型データを整える
# 
# __目次：__
# - <a href="#overview">データセットの概要</a>
# - <a href="#stim_check">入力パターン（視覚刺激）を調べる</a>
# - <a href="#stim_ds">動画の空間的なダウンサンプリング</a>
# - <a href="#resp_check">応答（BOLD信号）を調べる</a>
# - <a href="#cleanidx_tr">応答信号の前処理について</a>
# 
# ___
# 
# ここでは2種類のデータを扱うことになる。視覚刺激としての自然動画像と、fMRIで測定したBOLD信号（血中酸素濃度に依存する信号）の2種類である。前者はいわば「入力」で、後者は脳活動の指標で、「応答」である。
# 
# <img src="img/stimuli_ex.png" alt="Stimuli Image" width="240" height="180" />
# 
# <img src="img/fMRI_example.png" alt="fMRI Image" width="240" height="180" />
# 
# 主要の目的として、入力画像から脳活動を正確に予測することを掲げる。イメージとして、画像を応答信号へと「符号化」していることから、このシステムを「エンコーダ」と呼ぶことが多い。

# <a id="overview"></a>
# ## データセットの概要
# 
# 今回使うデータの名称は「Gallant Lab Natural Movie 4T fMRI Data set」で、通称は*vim-2*である。その中身を眺めてみると：
# 
# __視覚刺激__
# ```
# Stimuli.tar.gz 4089110463 (3.8 GB)
# ```
# 
# __BOLD信号__
# ```
# VoxelResponses_subject1.tar.gz 3178624411 (2.9 GB)
# VoxelResponses_subject2.tar.gz 3121761551 (2.9 GB)
# VoxelResponses_subject3.tar.gz 3216874972 (2.9 GB)
# ```
# 
# 視覚刺激はすべて`Stimuli.mat`というファイルに格納されている（形式：Matlab v.7.3）。その中身は訓練・検証データから構成される。
# 
#  - `st`: training stimuli. 128x128x3x108000 matrix (108000 128x128 rgb frames). 
#  - `sv`: validation stimuli. 128x128x3x8100 matrix (8100 128x128 rgb frames).
# 
# 訓練データに関しては、視覚刺激は15fpsで120分間提示されたため、7200の時点で合計108000枚のフレームから成る。検証データについては、同様に15fpsで9分間提示されたため、540の時点で合計8100枚のフレームから成る。検証用の視覚刺激は10回被験者に提示されたが、今回使う応答信号は、その10回の試行から算出した平均値である。平均を取る前の「生」データは公開されているが、ここでは使わない。
# 
# あと、データを並べ替える必要は特にない。作者の説明：
# 
# > *"The order of the stimuli in the "st" and "sv" variables matches the order of the stimuli in the "rt" and "rv" variables in the response files."*

# 前へ進むには、これらのファイルを解凍しておく必要がある。
# 
# ```
# $ tar -xzf Stimuli.tar.gz
# $ tar -xzf VoxelResponses_subject1.tar.gz
# $ tar -xzf VoxelResponses_subject2.tar.gz
# $ tar -xzf VoxelResponses_subject3.tar.gz
# ```
# 
# すると`Stimuli.mat`および`VoxelResponses_subject{1,2,3}.mat`が得られる。階層的な構造を持つデータなので、開閉、読み書き、編集等を楽にしてくれる__PyTables__(http://www.pytables.org/usersguide/index.html)というライブラリを使う。シェルからファイルの中身とドキュメンテーションを照らし合わせてみると、下記のような結果が出てくる。
# 
# ```
# $ ptdump Stimuli.mat
# / (RootGroup) ''
# /st (EArray(108000, 3, 128, 128), zlib(3)) ''
# /sv (EArray(8100, 3, 128, 128), zlib(3)) ''
# ```
# かなり単純な「階層」ではあるが、RootGroupにはフォルダー（`st`と`sv`）が2つある。それぞれの座標軸の意味を確認すると、1つ目は時点、2つ目は色チャネル（RGB）、3つ目と4つ目のペアは2次元配列における位置を示す。
# 
# 次に応答信号のデータに注視すると、もう少し豊かな階層構造が窺える。
# 
# ```
# $ ptdump VoxelResponses_subject1.mat 
# / (RootGroup) ''
# /rt (EArray(73728, 7200), zlib(3)) ''
# /rv (EArray(73728, 540), zlib(3)) ''
# /rva (EArray(73728, 10, 540), zlib(3)) ''
# (...Warnings...)
# /ei (Group) ''
# /ei/TRsec (Array(1, 1)) ''
# /ei/datasize (Array(3, 1)) ''
# /ei/imhz (Array(1, 1)) ''
# /ei/valrepnum (Array(1, 1)) ''
# /roi (Group) ''
# /roi/FFAlh (EArray(18, 64, 64), zlib(3)) ''
# /roi/FFArh (EArray(18, 64, 64), zlib(3)) ''
# /roi/IPlh (EArray(18, 64, 64), zlib(3)) ''
# /roi/IPrh (EArray(18, 64, 64), zlib(3)) ''
# /roi/MTlh (EArray(18, 64, 64), zlib(3)) ''
# /roi/MTplh (EArray(18, 64, 64), zlib(3)) ''
# /roi/MTprh (EArray(18, 64, 64), zlib(3)) ''
# /roi/MTrh (EArray(18, 64, 64), zlib(3)) ''
# /roi/OBJlh (EArray(18, 64, 64), zlib(3)) ''
# /roi/OBJrh (EArray(18, 64, 64), zlib(3)) ''
# /roi/PPAlh (EArray(18, 64, 64), zlib(3)) ''
# /roi/PPArh (EArray(18, 64, 64), zlib(3)) ''
# /roi/RSCrh (EArray(18, 64, 64), zlib(3)) ''
# /roi/STSrh (EArray(18, 64, 64), zlib(3)) ''
# /roi/VOlh (EArray(18, 64, 64), zlib(3)) ''
# /roi/VOrh (EArray(18, 64, 64), zlib(3)) ''
# /roi/latocclh (EArray(18, 64, 64), zlib(3)) ''
# /roi/latoccrh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v1lh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v1rh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v2lh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v2rh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v3alh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v3arh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v3blh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v3brh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v3lh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v3rh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v4lh (EArray(18, 64, 64), zlib(3)) ''
# /roi/v4rh (EArray(18, 64, 64), zlib(3)) ''
# 
# ```
# 
# 応答のデータでは、RootGroupのなかには、まず`rt`、`rv`、`rva`という3つの配列がある。これらはBOLD信号の測定値を格納している。また、`roi`と`ei`と名付けられたgroupがある。前者は応答信号の時系列とボクセルを結びつけるためのインデックスである。後者は実験の条件等を示す数値が格納されている。ここで`roi`のほうに注目すると、計測された脳の領域全体を分割して（構成要素：ボクセル）、それを生理的・解剖学的に関心を持つべき「関心領域」（ROI）に振り分けていくのだが、この`roi`なるグループは、各ROIの名が付いた配列を含む。たとえば、`v4rh`とは__V4__という領域で、左半球（__l__eft __h__emisphere）に限定したROIのことである。明らかなように、ROIの数はボクセル数（$18 \times 64 \times 64 = 73728$）よりも遥かに少ないので、各ROIが多数のボクセルを含むことがわかる。

# <a id="stim_check"></a>
# ## 入力パターン（視覚刺激）を調べる
# 
# それでは、蓋を開けて画像データを見てみよう。

# In[1]:



import numpy as np
import tables
import matplotlib
from matplotlib import pyplot as plt
import pprint as pp

# Open file connection.
f = tables.open_file("data/vim-2/Stimuli.mat", mode="r")

# Get object and array.
stimulus_object = f.get_node(where="/", name="sv")
print("stimulus_object:")
print(stimulus_object)
print(type(stimulus_object))
print("----")

stimulus_array = stimulus_object.read()
print("stimulus_array:")
#print(stimulus_array)
print(type(stimulus_array))
print("----")

# Close the connection.
f.close()

# Check that it is closed.
if not f.isopen:
    print("Successfully closed.")
else:
    print("File connection is still open.")


# 指定した配列の中身が予想通りのものを格納していること、正しく読み込めていることなどを確かめる必要がある。

# In[2]:


# Open file connection.
f = tables.open_file("data/vim-2/Stimuli.mat", mode="r")

# Get object and array.
stimulus_object = f.get_node(where="/", name="sv")
print("stimulus_object:")
print(stimulus_object)
print(type(stimulus_object))
print("----")

stimulus_array = stimulus_object.read()
print("stimulus_array:")
#print(stimulus_array)
print("(type)")
print(type(stimulus_array))
print("(dtype)")
print(stimulus_array.dtype)
print("----")

num_frames = stimulus_array.shape[0]
num_channels = stimulus_array.shape[1]
frame_w = stimulus_array.shape[2]
frame_h = stimulus_array.shape[3]

frames_to_play = 5

oneframe = np.zeros(num_channels*frame_h*frame_w, dtype=np.uint8).reshape((frame_h, frame_w, num_channels))
im = plt.imshow(oneframe)

for t in range(frames_to_play):
    oneframe[:,:,0] = stimulus_array[t,0,:,:] # red
    oneframe[:,:,1] = stimulus_array[t,1,:,:] # green
    oneframe[:,:,2] = stimulus_array[t,2,:,:] # blue
    plt.imshow(oneframe)
    plt.show()

f.close()


# 動画としての中身を視認するために、このJupyterノートブックよりも、シェルからPythonスクリプトを実行したほうが楽である：
# 
# ```
# $ cd scripts
# $ python vidcheck1.py
# ```
# わざわざコマンドラインを起動しているのは、matplotlibのバグ( https://github.com/matplotlib/matplotlib/pull/9415 )があるからである。
# 
# 十分なフレーム数を重ねてみると、読み込んだデータが確かに動画像であることが確認できた。但し、向きがおかしいことと、フレームレートがかなり高いこと、この2点は後ほど対応する。
# 
# 
# ### 練習問題 (A):
# 
# 0. 座標軸を適当に入れ替えて(`help(np.swapaxes)`を参照)、動画の向きを修正してみてください。
# 
# 0. 上では最初の数枚のフレームを取り出して確認したのだが、次は最後の数枚のフレームを確認しておくこと。
# 
# 0. 上記は検証データであったが、同様な操作（取り出して可視化すること）を訓練データ`st`に対しても行うこと。
# 
# 0. 上記の操作で、最初の100枚のフレームを取り出してください。フレームレートが15fpsなので、7秒弱の提示時間に相当する。上記のコードの`for`ループに着目し、`range`に代わって適当に`np.arange`を使うことで、時間的なダウンサンプリングが容易にできる。15枚につき1枚だけ取り出して、1fpsに変えてから、最初の100枚を見て、変更前との違いを確認してください。
# ___

# <a id="stim_ds"></a>
# ## 動画の空間的なダウンサンプリング
# 
# 画素数が多すぎると、後の計算が大変になるので、空間的なダウンサンプリング、つまり画像を縮小することが多い。色々なやり方はあるが、ここでは__scikit-image__というライブラリの`transform`モジュールから、`resize`という関数を使う。まずは動作確認。

# In[3]:


from skimage import transform as trans
import imageio

im = imageio.imread("img/bishop.png") # a 128px x 128px image

med_h = 96 # desired height in pixels
med_w = 96 # desired width in pixels
im_med = trans.resize(image=im, output_shape=(med_h,med_w), mode="reflect")

small_h = 32 # desired height in pixels
small_w = 32 # desired width in pixels
im_small = trans.resize(image=im, output_shape=(small_h,small_w), mode="reflect")

tiny_h = 16 # desired height in pixels
tiny_w = 16 # desired width in pixels
im_tiny = trans.resize(image=im, output_shape=(tiny_h,tiny_w), mode="reflect")

myfig = plt.figure(figsize=(18,4))

ax_im = myfig.add_subplot(1,4,1)
plt.imshow(im)
plt.title("Original image")
ax_med = myfig.add_subplot(1,4,2)
plt.imshow(im_med)
plt.title("Resized image (Medium)")
ax_small = myfig.add_subplot(1,4,3)
plt.imshow(im_small)
plt.title("Resized image (Small)")
ax_small = myfig.add_subplot(1,4,4)
plt.imshow(im_tiny)
plt.title("Resized image (Tiny)")

plt.show()


# どうも予想するように動いているようなので、視覚刺激の全フレームに対して同様な操作を行う。

# In[4]:


# Open file connection.
f = tables.open_file("data/vim-2/Stimuli.mat", mode="r")

# Get object and array.
stimulus_object = f.get_node(where="/", name="sv")
stimulus_array = stimulus_object.read()
num_frames = stimulus_array.shape[0]
num_channels = stimulus_array.shape[1]

# Swap the axes.
print("stimulus array (before):", stimulus_array.shape)
stimulus_array = np.swapaxes(a=stimulus_array, axis1=0, axis2=3)
stimulus_array = np.swapaxes(a=stimulus_array, axis1=1, axis2=2)
print("stimulus array (after):", stimulus_array.shape)

# Downsampled height and width.
ds_h = 96
ds_w = 96

stimulus_array_ds = np.zeros(num_frames*num_channels*ds_h*ds_w,                             dtype=np.float32).reshape((ds_h, ds_w, num_channels, num_frames))

for t in range(num_frames):
    stimulus_array_ds[:,:,:,t] = trans.resize(image=stimulus_array[:,:,:,t],
                                              output_shape=(ds_h,ds_w),
                                              mode="reflect")
    if t % 500 == 0:
        print("Update: t =", t)
        
f.close()


# ここで一つ注意すべきことは、縮小したあとの画像は、$\{0,1,\ldots,255\}$ではなく、$[0,1]$の実数値を取ることである。そのため、`dtype`を`np.float32`に変えている。

# In[5]:


print("(Pre-downsize) max:", np.max(stimulus_array),
      "min:", np.min(stimulus_array),
      "ave:", np.mean(stimulus_array))
print("(Post-downsize) max:", np.max(stimulus_array_ds),
      "min:", np.min(stimulus_array_ds),
      "ave:", np.mean(stimulus_array_ds))


# バイナリ形式にして、ディスクに保存しておくことにする（PyTablesと比べて、速いかどうかは各自で試すと良い）。

# In[6]:


fname = "data/vim-2/X_te.dat"
dtype = stimulus_array_ds.dtype
shape = stimulus_array_ds.shape
with open(fname, mode="bw") as fbin:
    stimulus_array_ds.tofile(fbin)
    print("Saved to file.")


# 読み書きが正しくできていることを確認すべく、最初の数枚をディスクから読み込んで、表示する。

# In[7]:



# Wipe the downsampled array.
stimulus_array_ds = np.zeros(num_frames*num_channels*ds_h*ds_w,                             dtype=np.float32).reshape((ds_h, ds_w, num_channels, num_frames))

# Load up the stored array.
with open(fname, mode="br") as fbin:
    print("Reading...", end=" ")
    stimulus_array_ds = np.fromfile(file=fbin, dtype=dtype).reshape(shape)
    print("OK.")

# Check a few frames.
num_frames = stimulus_array_ds.shape[3]

frames_to_play = 5

for t in range(frames_to_play):
    plt.imshow(stimulus_array_ds[:,:,:,t])
    plt.show()


# 
# ### 練習問題 (B):
# 
# 0. 上記の空間的ダウンサンプリングを訓練データ(`st`）に対しても行ない、`X_tr`として保存すること（先ほどの`X_te`と対をなす）。
# 
# 0. 後の解析のために、思い切って縮小した視覚刺激を用意すること。たとえば、$32 \times 32$ピクセルという程度である。名称として、`X_tr_32px` と`X_tr_32px`を使う。より少ない画素数からできている入力を使うことの利点は何か。その反面、どのようなデメリットがあるか。
# 
# ___
# 

# <a id="resp_check"></a>
# ## 応答（BOLD信号）を調べる
# 
# （カーネルをリセットした上で）次は応答のほうに主眼を置いて、調べていく。

# In[8]:


import numpy as np
import tables
import matplotlib
from matplotlib import pyplot as plt
import pprint as pp

# Open file connection.
f = tables.open_file("data/vim-2/VoxelResponses_subject1.mat", mode="r")

# Get objects and arrays.

response_object = f.get_node(where="/", name="rv")
idx_object = f.get_node(where="/roi/", name="v4lh")

print("response_object:")
print(response_object)
print(type(response_object))
print("----")
print("idx_object:")
print(idx_object)
print(type(idx_object))
print("----")

response_array = response_object.read()
idx_array = idx_object.read()

print("response_array:")
#print(response_array)
print(type(response_array))
print("----")
print("idx_array:")
#print(idx_array)
print(type(idx_array))
print("----")

# Close the connection.
f.close()

# Check that it is closed.
if not f.isopen:
    print("Successfully closed.")
else:
    print("File connection is still open.")


# 信号自体を見る前に、ボクセルとの対応関係を示すインデックスを開いてみる。

# In[9]:



pp.pprint(idx_array[0:2, 0:5,0:5])
print("dtype:", idx_array.dtype)
print("unique:", np.unique(idx_array))
print("sum =", np.sum(idx_array))


# 上記で明らかなように、これらの要素を全部足すことで、各ROIに含まれるボクセルの数がわかる。
# 
# ### 練習問題 (C):
# 
# 0. `np.nonzero`を使って、「活き」のインデックス（添字そのもの）を出すこと。
# 
# 0. V4というROIで、左半球においてボクセルが何個あるか。
# 
# 0. 左右合わせて、V4にボクセルが何個あるか。
# 
# 0. この数は、どの被験者でも共通しているか。
# 
# 0. 事前情報として、全部でボクセルが73728個あることはわかっている。これらのインデックスをすべて足し合わせて、期待通りのボクセル数になることを確認すること。
# 
# 0. 各ROIのボクセル数を棒グラフにして、表示すること(`help(plt.bar)`を参照)。ROIによって違うのであれば、どの領域がもっとも大きいか。もっとも小さいのはどれか。左右でボクセル数が違う領域はあるか。
# ___

# インデックスの働きと有用性は概ねわかってきたので、応答信号を見てみることにする。

# In[10]:


# Open file connection.
with tables.open_file("data/vim-2/VoxelResponses_subject1.mat", mode="r") as f:
    
    idx_object_myroi = f.get_node(where="/roi/", name="v4lh")
    idx_array_myroi = idx_object_myroi.read()
    
    indices = np.nonzero(idx_array_myroi.flatten())[0] # returns a tuple; extract the array.
    response_myroi = np.take(a=response_array, axis=0, indices=indices)
    print("shape:", response_myroi.shape)
    print("sum of index:", np.sum(idx_array_myroi))
    
num_secs = 120
time_idx = np.arange(0, num_secs, 1)

myfig = plt.figure(figsize=(10,8))

myfig.add_subplot(4,1,1)
plt.title("BOLD signal response from a handful of voxels")
val = response_myroi[0,time_idx]
plt.plot(time_idx, val)
print("dtype:", val.dtype)
print("num of nans:", np.sum(np.isnan(val)))

myfig.add_subplot(4,1,2)
val = response_myroi[1,time_idx]
plt.plot(time_idx, val)
print("dtype:", val.dtype)
print("num of nans:", np.sum(np.isnan(val)))

myfig.add_subplot(4,1,3)
val = response_myroi[2,time_idx]
plt.plot(time_idx, val)
print("dtype:", val.dtype)
print("num of nans:", np.sum(np.isnan(val)))

myfig.add_subplot(4,1,4)
val = response_myroi[3,time_idx]
plt.plot(time_idx, val)
print("dtype:", val.dtype)
print("num of nans:", np.sum(np.isnan(val)))

plt.xlabel("Elapsed time (s)")
plt.show()


# 動きは激しいが、一定の相関も見受けられるし、正しく読み込めているようには見える。正確に読めているならば、あとは欠落データに対応するだけだ。
# 
# 下記のように、欠損値のあるボクセルはすぐに見つかる。

# In[11]:



num_voxels = response_array.shape[0]
cleanidx = np.zeros((num_voxels,), dtype=np.uint32)

# Loop over the response array.
for v in range(num_voxels):
    
    if np.sum(np.isnan(response_array[v,:])) == 0:
        cleanidx[v] = 1 # on if the voxel is clean, otherwise off.

num_good = np.sum(cleanidx)
num_bad = np.int(num_voxels-num_good)
print("total good =", num_good, ";", format(100*num_good/num_voxels, ".2f"), "percent.")
print("total bad =", num_bad, ";", format(100*num_bad/num_voxels, ".2f"), "percent.")

# Print a simple bar graph.
plt.bar(np.arange(2), [num_good, num_bad])
plt.xticks(np.arange(2), ('Clean voxels', 'Bad voxels'))
plt.show()

with open("data/vim-2/cleanidx_te.dat", mode="bw") as fbin:
    cleanidx.tofile(fbin)
    print("Saved to file.")
    


# 一定の割合でデータが欠落しているのだが、幸いにもほとんどのボクセルは綺麗に取れている。
# 
# ### 練習問題 (D):
# 
# 0. `v1lh`と`v4lh`、それぞれの最初の4個のボクセルを取り出して、最初の120秒間の数値を上記のようにプロットすること。欠落データは見つかったか（欠損値として`nan`が使われている）。
# 
# 0. 訓練データにおいて、欠落データを含むボクセルが何個あるか。ROIごとに、欠落ボクセルの数を調べて、棒グラフで表示すること。欠落データが比較的に多い・少ない領域があるか。また、欠損値について、左右半球の違いは見られるか。
# 
# 0. 先ほどの`clean_te`と同様に、訓練データに対して、綺麗なボクセルのインデックスを調べ、`cleanidx_tr`として保存すること。また、最終的には訓練・検証ともにクリーンなボクセルだけを見ることになるであろうから、この2つの配列の共通部分を出して、`cleanidx`として保存すること。
# 
# ___
# 
# 「入力」としての視覚刺激と同様に、「出力」としてのBOLD信号もPythonのバイナリ形式でディスクに書き込むことにする。

# In[5]:



fname = "data/vim-2/y_te.dat"
dtype = response_array.dtype
shape = response_array.shape
with open(fname, mode="bw") as fbin:
    response_array.tofile(fbin)
    print("Saved to file.")
    


# 例のごとく、再構成を試みて、読み書きが正常にできているかどうかチェックする。

# In[6]:


# Wipe the response array.
response_array = np.zeros(response_array.size, dtype=response_array.dtype).reshape(response_array.shape)

# Load up the stored array.
with open(fname, mode="br") as fbin:
    print("Reading...", end=" ")
    response_array = np.fromfile(file=fbin, dtype=dtype).reshape(shape)
    print("OK.")

# Do the exact same visualization, and see if things change.    
num_secs = 120
time_idx = np.arange(0, num_secs, 1)

myfig = plt.figure(figsize=(10,8))

myfig.add_subplot(4,1,1)
plt.title("BOLD signal response from a handful of voxels")
val = response_myroi[0,time_idx]
plt.plot(time_idx, val)
print("dtype:", val.dtype)
print("num of nans:", np.sum(np.isnan(val)))

myfig.add_subplot(4,1,2)
val = response_myroi[1,time_idx]
plt.plot(time_idx, val)
print("dtype:", val.dtype)
print("num of nans:", np.sum(np.isnan(val)))

myfig.add_subplot(4,1,3)
val = response_myroi[2,time_idx]
plt.plot(time_idx, val)
print("dtype:", val.dtype)
print("num of nans:", np.sum(np.isnan(val)))

myfig.add_subplot(4,1,4)
val = response_myroi[3,time_idx]
plt.plot(time_idx, val)
print("dtype:", val.dtype)
print("num of nans:", np.sum(np.isnan(val)))

plt.xlabel("Elapsed time (s)")
plt.show()


# もとの階層型データから読み込んだものとまったく同じように見える（はずである）。
# 
# ### 練習問題 (E):
# 
# 0. 先ほど`rv`を使って`y_te`を作ったのと同様に、`rt`を使って`y_tr`を用意すること。
# 
# 0. もう少しだけ可視化のウォーミングアップをするため、8個のプロットから成るフィギュアを、次の仕様にしたがって、作成すること。並べ方としては、4x2（縦4個、横2）とする。左の列には、左半球からのデータを使い、右の列には右半球のデータを使う。プロットするデータの中身だが、欠損値の無いボクセルをV1から、4個選び、最初の60秒間の計測値を表示すること。各行はボクセルと対応付ける（4行、4個のボクセル）。
# 
# 0. 最後に、これまでは一人の被験者の分しか見てこなかったが、上記の一連の作業（Pythonバイナリ形式としての書き込み、綺麗なインデックスの作成など）を全被験者分、行うこと。名づけ方を少し変える必要があるため、たとえば、`sub1`、`sub2`、`sub3`などを最後につけることで明示できる。`y_tr_sub1.dat`、`y_tr_sub2.dat`、`y_tr_sub3.dat`といった具合である。`X`も`cleanidx`も同様に用意すること。
# 
# ___

# <a id="resp_pproc"></a>
# ## 応答信号の前処理について
# 
# 末筆ながら、BOLD信号には一定の前処理が必要である。非専門家にはとてもできない作業なので、ありがたいことにデータの作者は前処理されたデータを提供してくれている。これを確かめるために、ドキュメンテーションを読むことにする。*vim-2*の「dataset description」ファイルから：
# 
# > *"The functional data were collected for three subjects, in three sessions over three separate days for each subject (please see the Nishimoto et al. 2011 for scanning and preprocessing parameters). Peak BOLD responses to each timepoint in the stimuli were estimated from the preprocessed data."*
# 
# つまり、応答信号をいじる必要は特にないのである。実際にどのような前処理が行われてきたか知るためには、引用している論文(Nishimoto et al., 2011）の付録（*Supplemental Experimental Procedures -- Data Preprocessing*）では、「Peak BOLD responses」を算出するための手法がある程度は説明されている：
# 
# > *"BOLD signals were preprocessed as described in earlier publications. Briefly, motion compensation was performed using SPM '99, and supplemented by additional custom algorithms. For each 10 minute run and each individual voxel, drift in BOLD signals was first removed by fitting a third-degree polynomial, and signals were then normalized to mean 0.0 and standard deviation 1.0."*
# 
# 繰り返しにはなるが、この「Peak BOLD responses」の推定は、上記の前処理を経て、出されたものである。実際に提供されているデータの10分ブロックが正規化されているというわけではない。この点だけは注意すべきである。この事実は下記のようにすぐに確認できるが、「ほぼ」正規化されていることもわかる。

# In[9]:



# Load up the training data.
fname = "data/vim-2/y_tr.dat"
with open(fname, mode="br") as fbin:
    print("Reading...", end=" ")
    response_array = np.fromfile(file=fbin, dtype=np.float32).reshape((73728, 7200))
    print("OK.")
    
fname = "data/vim-2/cleanidx_tr.dat"
with open(fname, mode="br") as fbin:
    print("Reading...", end=" ")
    cleanidx_tr = np.fromfile(file=fbin, dtype=np.uint32)
    print("OK.")
    
response_array_clean = np.take(a=response_array, axis=0, indices=np.nonzero(cleanidx_tr)[0])

voxel_idx = 0 # the (clean) voxel idx to check
tmpspan = 600 # 10 minute "blocks"; units are seconds.
for i in range(response_array_clean.shape[1]//tmpspan):
    tmpi = i
    tmpidx = np.arange((tmpi*tmpspan), (tmpi+1)*tmpspan)
    response_tocheck = response_array_clean[voxel_idx,tmpidx]
    print("Block num =", i, ", mean =", np.mean(response_tocheck), ", std =", np.std(response_tocheck))


# ### 終わり： 後に使う関数などを`scripts/vim-2.py`に貼り付けること。

# ## 参考文献：
# 
#  - Nishimoto, Shinji, et al. "Reconstructing visual experiences from brain activity evoked by natural movies." Current Biology 21.19 (2011): 1641-1646.
#  - Description of dataset vim-2 (visual imaging 2), at CRCNS - Collaborative Research in Computational Neuroscience. https://crcns.org/data-sets/vc/vim-2/about-vim-2
