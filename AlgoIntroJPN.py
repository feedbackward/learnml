
# coding: utf-8

# # 学習アルゴリズムの基本的な設計
# 
# 学習機を実装する方法はいくらもあるが、教育を主要目的とするこのチュートリアルでは、基本的な役割に応じて、学習機を「__モデル__」と「__アルゴリズム__」という2つの要素に分けることにする。この役割についてざっくり述べると、
# 
#  - __モデル__： 学習用の訓練・検証データの情報を保持し、学習モデルを特徴づけるロス関数や勾配などを計算するメソッドを持つクラスオブジェクトである。
#  - __アルゴリズム__： これもまたクラスオブジェクトだが、イテレータでもある。初期値、モデルオブジェクト、そしてアルゴリズムの働きを細かく制御するパラメータを渡され、そのデータとモデルに応じて、パラメータを更新するための統計量と最適化を担う。
# 
# ここでは、モデルを抽象化したまま、アルゴリズムのほうに注目していく。
# 
# __目次：__
# 
# - <a href="#trivial">自明なアルゴリズムを実装する</a>
# - <a href="#nontrivial">非自明なアルゴリズムを実装する</a>
# - <a href="#testNoisyOpt">ノイジーな最適化：擬似データで学習則の働きを検証</a>
# - <a href="#GD">勾配情報を利用した更新方法</a>
# - <a href="#SGD">確率的なサブサンプリングを利用した更新方法</a>
# 
# ___

# <a id="trivial"></a>
# ## 自明なアルゴリズムを実装する
# 
# まずは単純な例から始める。仮に、学習課題の制約などで、ロス関数しか持っていないとする（例：一階微分が陽には表せない）。この状況下で、差分法を使った近似的な勾配降下法を実装していきたい。
# 
# 複雑な動きをするアルゴリズムを作る前に、下ごしらえとして「自明な」アルゴリズムを作って、より高度な手法の原型を覚えておく。

# In[1]:


class Algo_trivial:

    '''
    An iterator which does nothing special at all, but makes
    a trivial update to the initial parameters given, and
    keeps record of the number of iterations made.
    '''

    def __init__(self, w_init, t_max):
        
        self.w = np.copy(w_init)
        self.t_max = t_max


    def __iter__(self):

        self.t = 0
        print("(__iter__): t =", self.t)

        return self
    

    def __next__(self):

        # Condition for stopping.
        
        if self.t >= self.t_max:
            print("--- Condition reached! ---")
            raise StopIteration

        print("(__next__): t =", self.t)
        self.w += 5
        self.t += 1

        # Note that __next__ does not need to return anything.

    def __str__(self):

        out = "State of w:" + "\n" + "  " + str(self.w)
        return out


# クラスの定義自体は単純なのだが、実際に動かしてみたほうが更にわかりやすい。

# In[2]:



import numpy as np

al = Algo_trivial(w_init=np.array([1,2,3,4,5]), t_max=10)

print("Printing docstring:")
print(al.__doc__) # check the docstring (not)

for mystep in al:
    pass # do nothing special


# 特に何もしないアルゴリズムではあるが、反復演算がどのようになされるかはこれで明白に見えてくる。要点：
# 
#  - `StopIteration`という例外を投げると、ただちにループから脱出する。
#  
#  - 0番目のステップでは、`iter`と`next`は両方とも呼び出される。
# 
# なお、`for`ループを介さずして同じ演算をしようと思えば、下記のようにそれができる。

# In[3]:


iter(al)
next(al)
next(al)
next(al)
next(al) # and so on...


# In[4]:


al = Algo_trivial(w_init=np.array([1,2,3,4,5]), t_max=10)

for mystep in al:
    print(al) # useful for monitoring state.

print("One last check after exiting:")
print(al) # ensure that no additional changes were made.


# ### 練習問題 (A):
# 
# 0. `Algo_trivial`を反復するごとに、`w`がどのように更新されているか説明すること。この算法を「自明」と呼んでいるのは、データやモデルたるものには一切依存しない更新則だからである。
# 
# 0. 反復するごとに、`w`の全要素が倍増されるように`Algo_trivial`を改造してみること。
# 
# 0. さらに、初期値が$w=(0,1,2,3,4)$で、何回か倍増を繰り返したあとの状態が$w=(0, 16, 32, 48, 64) = (0 \times 2^4, 1 \times 2^4, 2 \times 2^4, 3 \times 2^4, 4 \times 2^4)$となるように、`al`を初期化する際に使うパラメータを設定すること。
# 
# 0. `al`を初期化しなおすこと無く、`for`ループを複数回走らせると、`w`の状態がどうなっていくか。
# ___

# <a id="nontrivial"></a>
# ## 非自明なアルゴリズムを実装する
# 
# さきほどの例では、初期状態とパラメータのみ渡して、固く決まった規則にしたがって更新している。実際には、データやモデルに応じて更新則を決めるだけの柔軟性を持たないと、アルゴリズムとしては役に立たない。次の事例では、望まれるような柔軟性を有するアルゴリズム設計を試みる。
# 
# 軽く定式化すると、ロス関すを$l(w;z)$と書く。最終的に決めたいパラメータとデータに依存するのである。このロス関数の下で「サンプル$\{l(w;z_{1}),\ldots,l(w;z_{n})\}$の平均を$w$について最小化する」という戦略でいくならば、よくある方法は勾配降下法である：
# 
# \begin{align}
# w_{(t+1)} = w_{(t)} - \alpha_{(t)} \frac{1}{n}\sum_{i=1}^{n} \nabla l(w_{(t)};z_{i}).
# \end{align}
# 
# しかし、我々の仮定としては、偏微分が求まらないという現状であるので、$\nabla l(\cdot;z)$は計算できない。そのときには、上記のアプローチではだめである。最も単純な代替案として、オイラーの頃から知られている「差分法」を使っての近似である。操作として、次元ごとにパラメータを微小な量だけ動かしたときの変化率を求める、それだけである。数式で表すと、
# 
# \begin{align}
# \Delta_{j} & = (0,\ldots,\delta,\ldots,0), \text{ such that} \\
# w + \Delta_{j} & = (w_{1},\ldots,(w_{j}+\delta),\ldots,w_{d}), \quad j = 1,\ldots,d 
# \end{align}
# 
# で動かして、関数のズレた分をパラメータのズレた分で割ると、偏微分の近似が得られる：
# 
# \begin{align}
# \widehat{g}_{i,j}(w) & = \frac{l(w+\Delta_{j};z_{i})-l(w;z_{i})}{\delta} \approx \frac{\partial l(w;z_{i})}{\partial w_{j}}, \quad i=1,\ldots,n \\
# \widehat{g}_{i}(w) & = \left(\widehat{g}_{i,1}(w),\ldots,\widehat{g}_{i,d}(w)\right), \quad i=1,\ldots,n.
# \end{align}
# 
# 勾配降下法の近似として、データ点ごとに偏微分の差分による近似を算出し、そのサンプル平均を真の勾配の代わりに使う。すると、更新式が以下のようになる：
# 
# \begin{align}
# w_{(t+1)} = w_{(t)} - \alpha_{(t)} \frac{1}{n}\sum_{i=1}^{n} \widehat{g}_{i}(w_{(t)}).
# \end{align}
# 
# この更新式を反映したアルゴリズムを次のように実装する。特に注目すべきは、`update`の中身だ。そこでロス関数を呼び出すから、自明でなくなるポイントでもある。

# In[5]:



class Algo_GD_FiniteDiff:

    '''
    Iterator which implements a line-search steepest descent method,
    via finite differences to approximate the gradient.
    '''

    def __init__(self, w_init, t_max, step, delta, verbose, store):

        # Store the user-supplied information.
        self.w = np.copy(w_init)
        self.t = None
        self.t_max = t_max
        self.step = step
        self.delmtx = np.eye(self.w.size) * delta
        self.delta = delta
        self.verbose = verbose
        self.store = store
        
        # If asked to store, keep record of all updates.
        if self.store:
            self.wstore = np.zeros((self.w.size,t_max+1), dtype=np.float64)
            self.wstore[:,0] = self.w.flatten()
        else:
            self.wstore = None
        

    def __iter__(self):

        self.t = 0

        if self.verbose:
            print("(via __iter__)")
            self.print_state()
        
        return self

    
    def __next__(self):

        # Condition for stopping.
        if self.t >= self.t_max:
            if self.verbose:
                print("--- Condition reached! ---")
            raise StopIteration

        self.t += 1

        if self.verbose:
            print("(via __next__)")
            self.print_state()


    def update(self, model):
        
        stepsize = self.step(self.t)
        newdir = np.zeros(self.w.size, dtype=self.w.dtype)
        loss = model.l_tr(self.w)

        # Perturb one coordinate at a time, compute finite difference.
        for j in range(self.w.size):
            
            # Perturb one coordinate. MODEL ACCESS here.
            delj = np.take(self.delmtx,[j],axis=1)
            loss_delta = model.l_tr((self.w + delj))
            
            newdir[j] = np.mean(loss_delta-loss) / self.delta
            
            
        self.w = self.w - stepsize * newdir.reshape(self.w.shape)
        
        if self.store:
            self.wstore[:,self.t] = self.w.flatten()


    def print_state(self):
        print("------------")
        print("t =", self.t, "( max = ", self.t_max, ")")
        print("w = ", self.w)
        print("------------")


# 見てもらうと明らかだが、`l_tr`の存在が重要である。これはモデルオブジェクトのメソッドであり、またデータにも依存する。モデルが出現するとはいえ、それは`model`という引数だけであって、その実態はまだ完全に抽象的である。`j`番目の反復ステップにおけるコードと数式の対応関係を表にまとめた：
# 
# | `code` | 数式 |
# | ------ | :----------------: |
# | `delta` | $\delta$ |
# | `delj` | $\Delta_{j}$ |
# | `loss` | $\left( l(w;z_{1}), \ldots, l(w;z_{n}) \right)$ |
# | `loss_delta` | $\left(l(w+\Delta_{j};z_{1}), \ldots, l(w+\Delta_{j};z_{n})\right)$ |
# | `newdir[j]` | $\sum_{i=1}^{n}\widehat{g}_{i,j}(w) / n$ |
# 
# 
# この`model`がどんなものなのか、まったく指定していない。重要なのは、実行時に何らかのモデルが、何らかのロスを計算してくれること、それだけである。かくてモデルとアルゴリズムの役割分担を実現し、それぞれの拡張性を高めることもできる。
# 
# 補足的な説明：
# 
#  - 新しく加わった引数： `step`、`delta`、`verbose`、`store`。
#  - 新しく加わったメソッド: `print_state`、`update`。
#  - `print_state`： 状態を知らせてくれるメソッド。
#  - `store`： Trueであれば、毎回反復するごとに得られるベクトルを$[w_{(0)}, \cdots, w_{(T)} ]$のように配列にまとめて格納する。この$T$が`t_max`に対応する。
#  - このやり方だと、反復している途中でも、モデルが変わることが可能である。換言すれば、設計者が望むなら、時間$t$において、$w_{(t)}$の状態に応じてモデルオブジェクトを変えることができる。変更されたモデルがアルゴリズムのメソッドに渡されると、$w_{(t+1)}$以降の更新に影響を及ぼすなど、等々。
# 
# 
# ### 練習問題 (B):
# 
# 上記の`FiniteDiff`の理解を確かめるために、下記の練習問題に答えてください。
# 
# 0. 関数`update`のなかで出てくる`for`では、座標軸を一つずつ扱っている。座標軸の数はいくらあるか。なぜ全部ではなく、一つずつ見ているか説明すること。
# 
# 0. 上の解説にもあったが、どのような学習課題だったら`FiniteDiff`を使うことが妥当っだと思われるか。具体例は必要としないが、もしあれば歓迎する。
# 
# 0. 上で何度も述べた「差分法」を使って、何を近似しようとしているか。記号を使って説明すること。より大きな`delta`を使うと、近似の精度が一般に良くなるか、それとも悪くなるか。
# 
# このアルゴリズムの動きを厳密に調べるためには、何らかのモデル（とデータ）を用意する必要がある。下記の例では、手作りの擬似データを使って、その働きを検証していく。
# ___

# <a id="testNoisyOpt"></a>
# ## ノイジーな最適化：擬似データで学習則の働きを検証
# 
# 自前のモデルを使ってアルゴリズムの挙動を調べる前に、そのモデルについて説明しておく。学習課題を「リスク最小化」としている。期待損失最小化ともいう。ロス関数を所与として、下記のように帰着する。
# 
# \begin{align}
# \min_{w \in \mathbb{R}^{d}} R(w), \quad R(w) = \mathbf{E}_{Z} l(w;z).
# \end{align}
# 
# ロス関数$l$は知っているが、いうまでもなく期待値を取るための確率分布、つまり「真の分布」は知らない。そのため、標本を使って近似するわけである。そのサンプルを$z_{1},\ldots,z_{n}$と表記する。
# 
# さて、自前のモデルでは、この$R$の形式を自ら下記のように定める：
# 
# \begin{align}
# R(w) = (w - w^{\ast})^{T}A(w - w^{\ast}) + b^2.
# \end{align}
# 
# そこから逆算して、条件$\mathbf{E}_{Z} l(w;z) = R(w)$がすべての$w \in \mathbb{R}^{d}$に対して満たされるように$l$を設計する。いくつかの攻め方はあるが、自然な例としては、線形回帰モデルの下での2乗誤差を考える。つまり、$z = (x,y)$とし、
# 
# \begin{align}
# y = \langle w^{\ast}, x \rangle + \varepsilon
# \end{align}
# 
# という過程で応答$y$が決まる。ノイズの項$\varepsilon$と入力$x$が独立であるとする。そのとき、$\mathbf{E}\varepsilon^2 = b^2$となって、また$\mathbf{E}\varepsilon x = 0$なので、展開して積分を取ると例の条件が満たされる。なお、$R$の定義式にある行列は下記のように決まる：
# 
# \begin{align}
# A = \mathbf{E}xx^{T}.
# \end{align}
# 
# この入力ベクトルが平均ゼロ（$\mathbf{E}_{X}x = 0$）であれば、$A =\text{cov}\,x$となる。このようにして$l$ も$R$もわかるようなスペシャルケースに着目するのは、学習アルゴリズムの汎化能力が正確に測定できるからである。
# 
# どのようなデータだったら従来の学習則が失敗するか、どのような改善策が学習効率に寄与するか、こういった問いかけに答えることが、この実験方法によって可能になる。

# 上で説明したモデルにしたがって、データをランダムに生成する関数をあらかじめ用意している。`support/parse_model`というディレクトリに保存されている`NoisyOpt_isoBig`である。ソースは自由に見てもらうことはできるが、ここではその実行結果だけを見ることにする。

# In[6]:


import support.parse_data as dp
import support.parse_model as mp
import pprint as pp

# Data information used to initialize the model.
data_info = dp.NoisyOpt_isoBig()

print("Data information:")
print(data_info)


# いくつか技術的な点：（`*_tr`が訓練データ(__tr__aining)で、`*_te`が検証データ(__te__sting)である）
# 
# - この一行を実行しただけで、データはすでに生成され、ディスクに保存されている。上記のコードを再実行するたびに、新しいデータが生成される。
# 
# ```
# $ ls -l data/NoisyOpt_isoBig
# ```
# 
# - この「data info」オブジェクトがデータに関する基本的な情報を格納する。たとえば`data_info.X_tr["path"]`には訓練データの入力のパスを保有している。
# 
# - 検証データの値が`None`となっているのは、検証するためのデータは要らないからである。学習結果の評価は、既知の$R$でできるのである。この関数を実装しているのは、`NoisyOpt`というモデルである。`mname`とは、このデータと相性の良いモデルを指名しているものである。
# 
# - `misc`では、モデル側で学習結果を評価する上で必要なパラメータを渡している。ここでは$R$を決めるためのパラメータである。対応関係：$\mathbf{E} \varepsilon^2$と`sigma_noise`、$\mathbf{E}xx^{T}$と`cov_X`、$w^{\ast}$と`w_true`。
# 
# `X_tr`の`shape`からわかるように、上記の規則にしたがってデータを生成した結果、$d=2$である。そうすることで、反復的に更新される学習アルゴリズムの状態の軌跡と、$R$の等高線も併せて可視化できる、という利点がある。ここからはモデルオブジェクトを作って、学習アルゴリズムを実際に働かせていく。

# In[7]:



mod = mp.model(data_info) # pass the data info.
print("Model information:")
print(mod)
print("checksum:", np.sum(mod.X_tr)) # if re-run the NoisyOpt_isoBig function, naturally data will change.

print("Attributes of the model:")
pp.pprint(dir(mod))


# モデルを初期化し、その属性を確認した。そのほとんどは気にする必要はないが、モデルを初期化したことで、ディスクからデータを読み込み、メモリーを使って格納していることは留意されたい：

# In[8]:


print(type(mod.X_tr), mod.X_tr.shape)
print(type(mod.y_tr), mod.y_tr.shape)


# さらには、`l_tr`というメソッドが見える。これは訓練データを基に、2乗誤差を計算してくれている。思い出してもらうと、このロスが`Algo_GD_FiniteDiff`では必要とされている。最後に`eval`を呼び出すことは$R(w)$を計算することに対応し、また`evalDist`を呼び出すことは$\|w-w^{\ast}\|$の計算に対応する。
# 
# あとはアルゴリズムのパラメータを用意することだ。特に、ステップサイズ$\alpha_{(t)}$が更新ステップに依存するため、コールバック関数を用意する必要がある。

# In[9]:


def alpha_fixed(t, val):
    '''
    Step-size function: constant.
    '''
    return val

def alpha_log(t, val=1):
    '''
    Step-size function: logarithmic.
    '''
    return val / (1+math.log((1+t)))

def alpha_pow(t, val=1, pow=0.5):
    '''
    Step-size function: polynomial.
    '''
    return val / (1 + t**pow)


# A function for making step-size functions.
def make_step(u):
    def mystep(t):
        return alpha_fixed(t=t, val=u)
    return mystep


# すべての準備が整ったので、擬似データを相手に、アルゴリズムを初期化して稼働させていこう。

# In[10]:


# Initial point.
w_init = np.array([0,0], dtype=np.float64).reshape((2,1))

# Initialize the algorithm object.
al = Algo_GD_FiniteDiff(w_init=w_init,                        delta=0.01,                        step=make_step(0.2),                        t_max=10,                        verbose=True,                        store=True)

al.print_state()


# 予想通りの初期化はできている。あとは終了条件が満たされるまで反復させるだけである。

# In[11]:



# Run the iterative procedure, using model-dependent "update" at each step.
for mystep in al:
    al.update(model=mod)
    
mypath = al.wstore


# 数値的な収束はもちろん確認しやすいが、今回は2次元平面の上で探索しているため、可視化するとその軌跡が一層わかりやすくなる。

# In[12]:


import matplotlib
import matplotlib.pyplot as plt



eval2D = np.vectorize(mod.eval2D_helper)
tmpdel = np.linalg.norm(mod.w_true-w_init) * 1
xvals = np.arange(mod.w_true[0]-tmpdel,mod.w_true[0]+tmpdel, 0.1)
yvals = np.arange(mod.w_true[1]-tmpdel,mod.w_true[1]+tmpdel, 0.1)
X, Y = np.meshgrid(xvals, yvals)
Z = eval2D(w1=X, w2=Y)

myfig = plt.figure(figsize=(6,6))
ax = myfig.add_subplot(1,1,1)
CS = ax.contour(X, Y, Z)
ax.quiver(mypath[0,:-1], mypath[1,:-1],
          mypath[0,1:]-mypath[0,:-1],
          mypath[1,1:]-mypath[1,:-1],
          scale_units='xy', angles='xy', scale=1, color='k')
CS.clabel(inline=1, fontsize=10)
ax.plot(*mod.w_true, 'r*', markersize=12) # print true value.
ax.plot(*mypath[:,-1], 'bo', markersize=6) # print our final estimate.
plt.title('Trajectory of finite-diff GD routine')
plt.show()

perf = mod.eval(al.w)
print("Error of final point:", perf)


# 少なくとも、我々の期待からは大きくそれた結果ではないはずである。完璧ではないが、確実に正解に近づいていることはわかる。下記の練習問題に際して、何度も上記のテストを再実行して、パラメータを変えることの影響を調べること。
# 
# ### 練習問題 (C):
# 
# 0. 初期状態`w_init`をより良く・悪くすることで、収束がどう変わるか（但し、$w^{\ast}=(\pi,e)$と固定）。
# 
# 0. 近似精度をつかさどる`delta`を変えることで、働きが大きく変わることはあるか。「安全な範囲」たるものがあれば、それがいくらぐらいか。
# 
# 0. 更新幅の`step`（＝$\alpha_{(t)}$）を変えると、アルゴリズムの挙動がどう変わるか。特に有効な方法があれば、その理由について説明すること。
# 
# 0. （おまけ）閾値パラメータ`thres`を追加し、新しい終了条件を加えること。つまり、$\|w_{(t+1)}-w_{(t)}\| \leq \epsilon$のときにただちに終了する。この$\epsilon$を指定するのは`thres`である。
# 
# 0. （おまけ）上記の更新則では、データ点ごとに差分法による近似を求めた__後に__サンプル平均をとっているが、その順番を逆にすることも可能である。つまり、ロスの平均をとってから、差分をとって偏微分の近似をする。性能として、この方法はどうなるか調べること。
# 
# 0. （おまけ）`support/parse_data.py`にある`NoisyOpt_isoBig`のソースコードを読み、ノイズの分布を変えてみてください。非ガウスの分布だと、アルゴリズムの性能がどう変わるのか。また、分散を大きく・小さくすることで、成績がどのように変わるか。
#  
# ___
# 
# 上の例では基本的な挙動が明白に見えてきたが、サンプルが及ぼす影響を見るためには、連続して複数のサンプルを取って実行したほうがわかりやすい。

# In[13]:



import support.parse_data as dp
import support.parse_model as mp
import pprint as pp

w_init = np.array([0,0], dtype=np.float64).reshape((2,1))

num_trials = 8

w_est_overtrials = {str(i):None for i in range(num_trials)} # initialize

for tr in range(num_trials):
    
    # Generate new data.
    data_info = dp.NoisyOpt_isoBig()
    mod = mp.model(data_info)
    
    # "Run" the algorithm.
    al = Algo_GD_FiniteDiff(w_init=w_init,                            delta=0.01,                            step=make_step(0.2),                            t_max=10,                            verbose=False,                            store=True)
    for mystep in al:
        al.update(model=mod)
        
    perf = mod.eval(al.w)
    print("Error:", perf)
    
    # Store the algorithm's output (the full trajectory).
    w_est_overtrials[str(tr)] = al.wstore
    

eval2D = np.vectorize(mod.eval2D_helper)
tmpdel = np.linalg.norm(mod.w_true-w_init) * 1
xvals = np.arange(mod.w_true[0]-tmpdel,mod.w_true[0]+tmpdel, 0.1)
yvals = np.arange(mod.w_true[1]-tmpdel,mod.w_true[1]+tmpdel, 0.1)
X, Y = np.meshgrid(xvals, yvals)
Z = eval2D(w1=X, w2=Y)
    
myfig = plt.figure(figsize=(10,20))

grididx = 1
for tr in range(num_trials):
    
    mypath = w_est_overtrials[str(tr)]
    
    ax = myfig.add_subplot(num_trials//2, 2, grididx)
    grididx += 1
    
    CS = ax.contour(X, Y, Z)
    ax.quiver(mypath[0,:-1], mypath[1,:-1],
              mypath[0,1:]-mypath[0,:-1],
              mypath[1,1:]-mypath[1,:-1],
              scale_units='xy', angles='xy', scale=1, color='k')
    CS.clabel(inline=1, fontsize=10)
    ax.plot(*mod.w_true, 'r*', markersize=12)
    ax.plot(*mypath[:,-1], 'bo', markersize=6)
    ax.axis("off")

plt.show()


# 上のプロットからわかる通り、毎回同一の初期値から始まるが、その軌跡と到達地点がサンプルのばらつきによって毎回異なる。サンプル数が小さくなること、ノイズの度合いが大きくなることなどで、試行間のばらつきが増加する。「近似的な最適化」として機械学習を理解することは、その本質に迫る王道であると考える。
# 
# 
# ### 練習問題 (D):
# 
# 0. `support/parse_data.py`のソースを開き、`NoisyOpt_isoBig`の定義を探しだしてください。サンプル数$n$を決める`n`を変えて、アルゴリズムの振る舞いに対する影響を調べること。大きな値（例：$n=1000$以上）にするとどうなるか。きわめて小さな$n$も試してみると良い。
# 
# 0. 今は`perf`に代入している成績をただ`print`しているだけだが、全試行分の成績を記録しておくように上記のコードを変えること。その成績の平均（`np.mean`）と標準偏差（`np.sd`）を求めること。
# 
# 0. （おまけ）上記の実装では、「forward difference」という差分法を使って$\nabla l(w;z)$を近似しているが、差分法にはいろいろな種類がある。たとえば、
# <br>
# \begin{align}
# \frac{l(w;z_{i})-l(w-\Delta_{j} ;z_{i})}{\delta} \quad \text{and} \quad \frac{l(w+\Delta_{j};z_{i})-l(w-\Delta_{j};z_{i})}{2\delta}
# \end{align}
# <br>
# はそれぞれ「backward difference」と「central difference」と呼ばれる差分法である。これらを使った`Algo_GD_FiniteDiff`の改良版を作って、その振る舞いを調べてみること。$n$が小さくなるとその優劣に差があるか。
# ___
# 

# <a id="GD"></a>
# ## 勾配情報を利用した更新方法

# 勾配が直接求められない状況について考えてきたが、次は勾配が手に入るという状況について考える。実際、よく使われるロス関数の多くは、偏微分が計算しやすく、勾配が容易に求まることが多い。たとえば、$z = (x,y) \in \mathbb{R}^{d+1}$として、線形モデルの下での2乗誤差$l(w;z) = (y - \langle w,x \rangle)^{2}$の勾配がすぐに計算できる：
# 
# \begin{align}
# \nabla l(w;z) = 2(y-\langle w,x \rangle)(-1)x.
# \end{align}
# 
# ニューラルネットワークを含むより複雑なモデルでも、勾配が求められることがあるので、そのときには勾配降下法の一種が重宝される。解析的に勾配が手に入るならば、計算資源をその近似に使うことは無駄で、差分法は不要になる。つまり、通常の更新則が下記のように使える：
# 
# \begin{align}
# w_{(t+1)} = w_{(t)} - \alpha_{(t)} \frac{1}{n}\sum_{i=1}^{n} \nabla l(w_{(t)};z_{i}).
# \end{align}
# 
# これを実装することは、先ほどのアルゴリズムを踏まえると至って簡単である。`Algo_GD`として実装してみよう。

# In[14]:


class Algo_GD(Algo_GD_FiniteDiff):

    '''
    Iterator which implements a line-search steepest descent method,
    using the sample mean estimate of the gradient.
    '''

    def __init__(self, w_init, t_max, step, verbose, store):
        
        super(Algo_GD,self).__init__(w_init=w_init, t_max=t_max, step=step,
                                     delta=0, verbose=verbose, store=store)


    def update(self, model):
        
        stepsize = self.step(self.t)
        
        # Instead of finite-difference approximation, simply
        # access the gradient method of our model object.
        newdir = np.mean(model.g_tr(w=self.w),
                         axis=0) # take sample mean to get update direction.
        
        self.w = self.w - stepsize * newdir.reshape(self.w.shape)
        
        if self.store:
            self.wstore[:,self.t] = self.w.flatten()


# これまでと同様に、抽象的な「モデル」オブジェクトをここでも使っている。やはり`model`である。今回違うのは、`l_tr`ではなく、`g_tr`を用いていることである。これは勾配を訓練データで計算するメソッドである。`g_tr`は寸法$n \times d$の配列である。$d$が$w$の長さで、その中身は：
# 
# \begin{align}
# \begin{bmatrix}
# \nabla l(w;z_{1})\\
# \vdots\\
# \nabla l(w;z_{n})
# \end{bmatrix}.
# \end{align}
# 
# 列ごとに算術平均を求めると、ロスのサンプル平均の勾配そのものが求まる。どのような成績を叩き出すか、調べていく。

# In[15]:



import support.parse_data as dp
import support.parse_model as mp
import pprint as pp

w_init = np.array([0,0], dtype=np.float64).reshape((2,1))

num_trials = 8

w_est_overtrials = {str(i):None for i in range(num_trials)} # initialize

for tr in range(num_trials):
    
    # Generate new data.
    data_info = dp.NoisyOpt_isoBig()
    mod = mp.model(data_info)
    
    # "Run" the algorithm.
    al = Algo_GD(w_init=w_init,                 step=make_step(0.2),                 t_max=10,                 verbose=False,                 store=True)
    for mystep in al:
        al.update(model=mod)
        
    perf = mod.eval(al.w)
    print("Error:", perf)
    
    # Store the algorithm's output (the full trajectory).
    w_est_overtrials[str(tr)] = al.wstore
    

eval2D = np.vectorize(mod.eval2D_helper)
tmpdel = np.linalg.norm(mod.w_true-w_init) * 1
xvals = np.arange(mod.w_true[0]-tmpdel,mod.w_true[0]+tmpdel, 0.1)
yvals = np.arange(mod.w_true[1]-tmpdel,mod.w_true[1]+tmpdel, 0.1)
X, Y = np.meshgrid(xvals, yvals)
Z = eval2D(w1=X, w2=Y)
    
myfig = plt.figure(figsize=(10,20))

grididx = 1
for tr in range(num_trials):
    
    mypath = w_est_overtrials[str(tr)]
    
    ax = myfig.add_subplot(num_trials//2, 2, grididx)
    grididx += 1
    
    CS = ax.contour(X, Y, Z)
    ax.quiver(mypath[0,:-1], mypath[1,:-1],
              mypath[0,1:]-mypath[0,:-1],
              mypath[1,1:]-mypath[1,:-1],
              scale_units='xy', angles='xy', scale=1, color='k')
    CS.clabel(inline=1, fontsize=10)
    ax.plot(*mod.w_true, 'r*', markersize=12)
    ax.plot(*mypath[:,-1], 'bo', markersize=6)
    ax.axis("off")

plt.show()


# ロスの勾配を使うことで、一部の誤差を解消することができるため、全体としては不確実性が減る。とはいえ、最大の不確実性の原因となるのは、ランダムサンプルごとのばらつきであることから、ロスの勾配を使っていても、最適な軌道から逸れてしまうことは多少は避けられない。
# 
# 
# ### 練習問題 (E):
# 
# 0. 前の事例と同様に、全試行分の成績（誤差）を記録するようにコードを変えること。平均と分散を計算し、表示すること。
# 
# 0. 差分法を使った場合と比べて、誤差の平均と分散がどうか。$\delta$のどの水準で`Algo_GD_FiniteDiff`と`Algo_GD`が肩を並べることになるか。また、前者が後者に負ける水準とはどの程度なのか。固定したステップサイズを大きく・小さくすることで、これらの水準がどう変わるか。
# 
# 勾配降下法やそのバリエーションを使う学習アルゴリズムが実に多い。その改造版の一つで、もっとも単純かつ広く使われているのが、確率的勾配降下法である。反復する際、ランダムに小さなサブサンプル（標本の部分集合）を取って、その小さなサンプルだけで勾配の平均を求めて更新していくという手法である。更新がかなりノイジーにはなるが、$nd$個の偏微分を計算する場合と比べると遥かにコストが安い。この手法を次に見ることにしよう。

# <a id="SGD"></a>
# ## 確率的なサブサンプリングを利用した更新方法

# 制御できるパラメータ$w=(w_{1},\ldots,w_{d})$の数が多いと、勾配を求めるためには多数の偏微分を計算しないといけない。標本数$n$も次元$d$も大きい場合は、この計算を何度も行なうことは現実的でない。先ほど述べたように、この計算的な負荷を避ける単純な方法として、ミニバッチというサブサンプルを反復するごとにランダムに選び、そのミニバッチ分の勾配だけ計算して勾配降下法の更新式に代入する、という方法だ。つまり、インデックス
# 
# \begin{align}
# \mathcal{I} \subset \{1,2,\ldots,n\}
# \end{align}
# 
# を無作為に選ぶ。その要素の数が$|\mathcal{I}| = B$であるとする。そのときの更新則が下記のとおりになる。
# 
# \begin{align}
# w_{(t+1)} = w_{(t)} - \alpha_{(t)} \frac{1}{B}\sum_{j \in \mathcal{I}} \nabla l(w_{(t)};z_{j}).
# \end{align}
# 
# 古典的には、$B=1$としてデータ点一つずつ使って更新していく。これを確率的勾配降下法（SGD）と呼ぶことが多い。基本的な挙動を理解する上で、もしミニバッチのデータ（各$i \in \mathcal{I}$）を一つずつ独立かつ一様に$\{1,\ldots,n\}$からサンプルしているのであれば、期待値をとると、任意の$w$に対して下記の式が成り立つ。
# 
# \begin{align}
# \mathbf{E} \left( \frac{1}{B}\sum_{i \in \mathcal{I}} \nabla l(w;z_{i}) \right) & = \frac{1}{B}\sum_{i \in \mathcal{I}} \mathbf{E} \nabla l(w;z_{i})\\
#  & = \frac{1}{B}\sum_{j \in \mathcal{I}}\left(\frac{1}{n} \sum_{i=1}^{n} \nabla l(w;z_{j}) \right)\\
#  & = \frac{1}{n} \sum_{i=1}^{n} \nabla l(w;z_{j}).
# \end{align}
# 
# つまり、平均的には、この確率変数が、すべてのデータ点を使った場合の勾配そのものになる。いうまでもなく、反復回数が安くなるので、十分な回数を重ねておけば、そのばらつきが平準化され、良い解を安価で手に入れることを期待する。実装し、その働きを確認してみよう。

# In[16]:


class Algo_SGD(Algo_GD_FiniteDiff):

    '''
    Iterator which implements a line-search steepest descent method,
    using the sample mean estimate of the gradient.
    '''

    def __init__(self, w_init, batchsize, t_max, step, delta, verbose, store):
        
        super(Algo_SGD,self).__init__(w_init=w_init, t_max=t_max, step=step,
                                      delta=delta, verbose=verbose, store=store)
        
        self.batchsize = batchsize


    def update(self, model):
        
        stepsize = self.step(self.t)
        
        # Instead of using the "full gradient" averaged over the
        # whole batch, use a randomly selected sub-sample.
        fullgrad = model.g_tr(w=self.w)
        shufidx = np.random.choice(model.n, size=self.batchsize, replace=False)
        minigrad = np.take(fullgrad,shufidx,axis=0)
        
        newdir = np.mean(minigrad,
                         axis=0) # take sample mean to get update direction.
            
        self.w = self.w - stepsize * newdir.reshape(self.w.shape)
        
        if self.store:
            self.wstore[:,self.t] = self.w.flatten()
        


# `Algo_GD`のときと同様に、`g_tr`というメソッドを使って、訓練データ上の勾配を計算する。今回違うのは、そのなかからランダムに部分集合をサンプルしていること。対応関係をここでも明示しておくと：
# 
# | `code` | 数式 |
# | ------ | :----------------: |
# | `shufidx` | $\mathcal{I}$ |
# | `batchsize` | $B$ |
# | `newdir` | $\sum_{i \in \mathcal{I}} \nabla l(w;z_{i}) / B$ |
# | `model.n` | $n$ |
# 
# ＊もちろん、この実装はきわめて非効率である。実際には$n$個の勾配ベクトルを毎回計算しているから、サブサンプリングの計算面のメリットが無い（モデルを少し改造することですぐに効率化できる）。それでもここではミニバッチを使うと状態の軌跡がどう変わるか見たいだけなのである。
# 
# このSGDを起動し、先ほどのアルゴリズムと同じ擬似データを与えてみることにする。

# In[17]:



import support.parse_data as dp
import support.parse_model as mp
import pprint as pp

w_init = np.array([0,0], dtype=np.float64).reshape((2,1))

num_trials = 8

w_est_overtrials = {str(i):None for i in range(num_trials)} # initialize

for tr in range(num_trials):
    
    # Generate new data.
    data_info = dp.NoisyOpt_isoBig()
    mod = mp.model(data_info)
    
    # "Run" the algorithm.
    al = Algo_SGD(w_init=w_init,                  batchsize=1,
                  delta=0.01,\
                  step=make_step(0.2),\
                  t_max=10,\
                  verbose=False,\
                  store=True)
    for mystep in al:
        al.update(model=mod)
        
    perf = mod.eval(al.w)
    print("Error:", perf)
    
    # Store the algorithm's output (the full trajectory).
    w_est_overtrials[str(tr)] = al.wstore
    

eval2D = np.vectorize(mod.eval2D_helper)
tmpdel = np.linalg.norm(mod.w_true-w_init) * 1
xvals = np.arange(mod.w_true[0]-tmpdel,mod.w_true[0]+tmpdel, 0.1)
yvals = np.arange(mod.w_true[1]-tmpdel,mod.w_true[1]+tmpdel, 0.1)
X, Y = np.meshgrid(xvals, yvals)
Z = eval2D(w1=X, w2=Y)
    
myfig = plt.figure(figsize=(10,20))

grididx = 1
for tr in range(num_trials):
    
    mypath = w_est_overtrials[str(tr)]
    
    ax = myfig.add_subplot(num_trials//2, 2, grididx)
    grididx += 1
    
    CS = ax.contour(X, Y, Z)
    ax.quiver(mypath[0,:-1], mypath[1,:-1],
              mypath[0,1:]-mypath[0,:-1],
              mypath[1,1:]-mypath[1,:-1],
              scale_units='xy', angles='xy', scale=1, color='k')
    CS.clabel(inline=1, fontsize=10)
    ax.plot(*mod.w_true, 'r*', markersize=12)
    ax.plot(*mypath[:,-1], 'bo', markersize=6)
    ax.axis("off")

plt.show()


# 
# ### 練習問題 (F):
# 
# 0. ミニバッチの大きさを$B=1$とする。`step`と`t_max`を固定したとき、上記3つのアルゴリズム（`FiniteDiff`、`GD`、`SGD`）の軌跡、優劣がどうなるのか。まず明らかなのは、SGDの動きの激しさである。
# 
# 0. 反復回数`t_max`を増やし、ステップサイズを小さくしてみてください。SGDの軌跡が安定化するか。正解に近づくように見えるか。
# 
# 0. $B=1$としたとき、`SGD`の誤差の平均と標準偏差が`GD`と比べて、どうか。反復回数を増やして、ステップサイズを小さくすることで「勝負」できるようになるか。また、$B$をどこまで増やせば`SGD`と`GD`の区別がつかなくなるか。
# 
# 0. 今の実装では、`np.random.choice`に`replace=False`を渡しているから、無作為に選ぶインデックス`shufidx`が非復元抽出である（sampling without replacement）。一様にすべての点を選ぶために、復元抽出に変えてみること。性能が変わるか。変わるなら、どのように変わるか説明すること。
# 
# 0. （おまく）今は`al.w`を使って、「成績」としての誤差を計算している。これは最終ステップにおける$w_{(t)}$であるが、これは当然ながら最後のステップのランダムな動きに対して敏感である。このばらつきを和らげるために、たとえば最後の5ステップ分の状態（`al.wstore`に格納）の平均を取って、その成績を調べてみること。これは、「$\alpha$-suffix averaging」の一例で、その有効性は詳しく解析されている(Rakhlin et al., 2012)。

# ### 終わり： 後に使う関数などを`scripts/AlgoIntro.py`に貼り付けること。

# ## 参考文献：
# 
# - P. Frey MA691 course notes, "The finite difference method". (https://www.ljll.math.upmc.fr/frey/ma691.html)
# - Rakhlin, Alexander, Ohad Shamir, and Karthik Sridharan. "Making Gradient Descent Optimal for Strongly Convex Stochastic Optimization." ICML. 2012. arXiv URL: https://arxiv.org/abs/1109.5647
