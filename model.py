import brainunit as u
import brainstate as bst
import brainpy as bp
import jax
import jax.numpy as jnp


class SNN_ext(bst.nn.DynamicsGroup):
    def __init__(self, num_in, num_rec, num_out, exc_ratio=0.8,
                 tau_neu=300 * u.ms, tau_syn=300 * u.ms, tau_out=300 * u.ms,
                 ff_scale=1., rec_scale=1., E_exc=3. * u.mV, E_inh=-3. * u.mV,
                 i2r_prob=0.5, r2r_prob=0.2, r2o_prob=0.1, seed=42,
                 spike_count_num=25, batch_size=40, window_size=100):
        # 初始化父类DynamicsGroup
        super(SNN_ext, self).__init__()

        # 参数定义
        self.num_in = num_in
        self.num_rec = num_rec
        self.num_exc = int(num_rec * exc_ratio)
        self.num_inh = num_rec - self.num_exc
        self.num_out = num_out

        # connections
        self.i2r_prob = jnp.array(bst.random.rand(self.num_in, self.num_rec) < i2r_prob)
        self.r2r_conn = jnp.array(bst.random.rand(self.num_rec, self.num_rec) < r2r_prob)
        self.r2o_conn = jnp.array(bst.random.rand(self.num_rec, self.num_out) < r2o_prob)

        # remove self-connections
        self.r2r_conn = self.r2r_conn & ~jnp.eye(self.num_rec, dtype=bool)

        # spilt into exc2r_conn, inh2r_conn, exc2o_conn, inh2o_conn
        self.exc2r_conn = self.r2r_conn[:self.num_exc, :]
        self.inh2r_conn = self.r2r_conn[self.num_exc:, :]
        self.exc2o_conn = self.r2o_conn[:self.num_exc, :]
        self.inh2o_conn = self.r2o_conn[self.num_exc:, :]

        # get spike counts
        self.spike_count_num = spike_count_num
        self.batch_size = batch_size
        self.window_size = window_size
        self.spike_counts = bst.State(jnp.zeros((self.spike_count_num, self.batch_size, self.num_rec)))
        self.temp_spike = bst.State(jnp.zeros((self.window_size, self.batch_size, self.num_rec)))
        self.temp_i = bst.State(0)
        self.spike_count_i = bst.State(0)

        # 定义从输入层到递归层的连接（突触: i->r）
        # 使用Sequential将线性层和指数衰减层连接在一起
        ff_init = bst.init.KaimingNormal(scale=7 * (1 - (u.math.exp(-bst.environ.get_dt(), unit_to_scale=u.ms))),
                                         unit=u.mA)
        self.i2r = bst.nn.Sequential(
            # 线性层：用于将输入信号映射到递归层
            bst.nn.Linear(
                num_in, num_rec,  # 从输入层到递归层的连接数
                w_init=ff_init,  # 使用Kaiming Normal初始化权重
                b_init=bst.init.ZeroInit(unit=u.mA),  # 偏置初始化为零
                w_mask=self.i2r_prob  # 使用概率掩码，控制连接的稀疏性
            ),
            # 指数衰减层：对信号进行时间上的衰减，使其符合生物神经元动力学
            bst.nn.Expon(num_rec, tau=tau_syn, g_initializer=bst.init.Constant(0. * u.mA))
        )

        # 定义递归层（r），采用LIF神经元模型
        self.r = bst.nn.LIF(
            num_rec,  # 递归层神经元数量
            tau=tau_neu,  # 时间常数，控制膜电位衰减速率
            V_reset=0 * u.mV,  # 膜电位复位值
            V_rest=0 * u.mV,  # 静息膜电位
            V_th=1. * u.mV,  # 膜电位阈值，超过此值时神经元发放脉冲
            spk_fun=bst.surrogate.ReluGrad()  # 近似求导函数，用于实现脉冲发放
        )

        # recurrent to recurrent
        self.exc2r = bst.nn.Sequential(
            bst.nn.Linear(
                self.num_exc, self.num_rec,
                w_init=bst.init._random_inits.Gamma(shape=0.5, scale=0.1, unit=u.mS),
                b_init=bst.init.ZeroInit(unit=u.mS),
                w_mask=self.exc2r_conn
            ),
            bst.nn.Expon(num_rec, tau=tau_syn, g_initializer=bst.init.Constant(0. * u.mS)),
        )
        self.exc2r_coba = bst.nn.COBA(E_exc)

        self.inh2r = bst.nn.Sequential(
            bst.nn.Linear(
                self.num_inh, self.num_rec,
                w_init=bst.init._random_inits.Gamma(shape=0.5, scale=0.1, unit=u.mS),
                b_init=bst.init.ZeroInit(unit=u.mS),
                w_mask=self.inh2r_conn
            ),
            bst.nn.Expon(num_rec, tau=tau_syn, g_initializer=bst.init.Constant(0. * u.mS)),
        )
        self.inh2r_coba = bst.nn.COBA(E_inh)

        # output
        self.o = bst.nn.Expon(
            num_out, tau=tau_out,
            g_initializer=bst.init.Constant(0.)
        )

        # recurrent to output
        self.exc2o = bst.nn.Linear(
            self.num_exc, self.num_out,
            w_init=bst.init.KaimingNormal(scale=rec_scale),
            # b_init=bst.init.ZeroInit(unit=u.mA),
            w_mask=self.exc2o_conn
        )

    # update方法：用于执行网络的一次更新，返回输出层的输出
    def update(self, spike, ext_current):
        rec_spikes = self.r.get_spike()
        e_sps, i_sps = jnp.split(rec_spikes, [self.num_exc], axis=-1)

        # print(f'rec_spikes: {rec_spikes.sum()}')
        self.update_spike_count(rec_spikes)

        i2r_current = self.i2r(spike)
        exc2r_current = self.exc2r_coba.update(self.exc2r(e_sps), self.r.V.value)
        inh2r_current = self.inh2r_coba.update(self.inh2r(i_sps), self.r.V.value)

        r_current = i2r_current + exc2r_current + inh2r_current
        r_current = r_current.at[:, :self.num_exc].set(r_current[:, :self.num_exc] + ext_current)
        self.r(r_current)

        o_current = self.exc2o(e_sps)

        return self.o(o_current), self.r.V.value.mantissa

    # predict方法：用于预测并获取递归层的膜电位值、脉冲输出和最终输出
    def predict(self, spike, ext_current):
        rec_spikes = self.r.get_spike()
        e_sps, i_sps = jnp.split(rec_spikes, [self.num_exc], axis=-1)

        self.update_spike_count(rec_spikes)

        i2r_current = self.i2r(spike)
        exc2r_current = self.exc2r_coba.update(self.exc2r(e_sps), self.r.V.value)
        inh2r_current = self.inh2r_coba.update(self.inh2r(i_sps), self.r.V.value)

        r_current = i2r_current + exc2r_current + inh2r_current
        r_current = r_current.at[:, :self.num_exc].set(r_current[:, :self.num_exc] + ext_current)
        self.r(r_current)

        o_current = self.exc2o(e_sps)


        out = self.o(o_current)

        # 返回递归层的膜电位值、递归层脉冲输出和最终输出
        return self.r.V.value, rec_spikes, out

    def get_weight_matrix(self):
        exc2r_weight = self.exc2r.layers[0].weight.value['weight'].mantissa
        inh2r_weight = self.inh2r.layers[0].weight.value['weight'].mantissa
        return jnp.concatenate([exc2r_weight, inh2r_weight], axis=0)

    def set_weight_matrix(self, r2r_weight):
        exc2r_weight = r2r_weight[:self.num_exc]
        inh2r_weight = r2r_weight[self.num_exc:]
        self.exc2r.layers[0].weight.value['weight'] = exc2r_weight * u.mS
        self.inh2r.layers[0].weight.value['weight'] = inh2r_weight * u.mS

    def start_spike_count(self):
        self.spike_counts.value = jnp.zeros((self.spike_count_num, self.batch_size, self.num_rec))
        self.temp_spike.value = jnp.zeros((self.window_size, self.batch_size, self.num_rec))
        self.temp_i.value = 0
        self.spike_count_i.value = 0

    def update_spike_count(self, spike):
        self.temp_spike.value = self.temp_spike.value.at[self.temp_i.value].set(spike)
        self.temp_i.value = self.temp_i.value + 1
        # jax.debug.print(f'temp_i: {self.temp_i}')
        bst.compile.cond(self.temp_i.value == self.window_size - 1, self.update_spike_count_i, lambda: None)
        # if self.temp_i.value == (self.window_size - 1):
        #     self.temp_i.value = 0
        #     jax.debug.print(f'spike_count_i: {self.spike_count_i}')
        #     self.spike_counts.value = self.spike_counts.value.at[self.spike_count_i.value].set(self.temp_spike.value.sum(axis=0))

    def update_spike_count_i(self):
        self.temp_i.value = 0
        # jax.debug.print(f'spike_count_i: {self.spike_count_i}')
        self.spike_counts.value = self.spike_counts.value.at[self.spike_count_i.value].set(self.temp_spike.value.sum(axis=0))
        self.spike_count_i.value = self.spike_count_i.value + 1

    def get_spike_counts(self):
        return self.spike_counts.value