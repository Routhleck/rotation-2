import brainstate as bst
import brainunit as u
import jax.numpy as jnp


class SNN_ext(bst.nn.DynamicsGroup):
    """
    脉冲神经网络(SNN)扩展模型
    包含输入层、递归层(兴奋性和抑制性神经元)和输出层
    """
    def __init__(self, num_in, num_rec, num_out, exc_ratio=0.8,
                 tau_neu=300 * u.ms, tau_syn=300 * u.ms, tau_out=300 * u.ms,
                 E_exc=3. * u.mV, E_inh=-3. * u.mV,
                 i2r_prob=0.5, r2r_prob=0.2, r2o_prob=0.1, 
                 spike_count_num=25, batch_size=40, window_size=100):
        """
        参数:
            num_in: 输入层神经元数量
            num_rec: 递归层神经元数量
            num_out: 输出层神经元数量
            exc_ratio: 兴奋性神经元比例
            tau_neu: 神经元时间常数
            tau_syn: 突触时间常数
            tau_out: 输出层时间常数
            E_exc: 兴奋性突触反转电位
            E_inh: 抑制性突触反转电位
            i2r_prob: 输入层到递归层连接概率
            r2r_prob: 递归层内部连接概率
            r2o_prob: 递归层到输出层连接概率
        """
        super(SNN_ext, self).__init__()

        # 网络结构参数
        self.num_in = num_in
        self.num_rec = num_rec
        self.num_exc = int(num_rec * exc_ratio)  # 兴奋性神经元数量
        self.num_inh = num_rec - self.num_exc    # 抑制性神经元数量
        self.num_out = num_out

        # 连接矩阵初始化
        self.i2r_prob = jnp.array(bst.random.rand(self.num_in, self.num_rec) < i2r_prob)
        self.r2r_conn = jnp.array(bst.random.rand(self.num_rec, self.num_rec) < r2r_prob)
        self.r2o_conn = jnp.array(bst.random.rand(self.num_rec, self.num_out) < r2o_prob)

        # 移除自连接
        self.r2r_conn = self.r2r_conn & ~jnp.eye(self.num_rec, dtype=bool)

        # 分离兴奋性和抑制性连接
        self.exc2r_conn = self.r2r_conn[:self.num_exc, :]
        self.inh2r_conn = self.r2r_conn[self.num_exc:, :]
        self.exc2o_conn = self.r2o_conn[:self.num_exc, :]
        self.inh2o_conn = self.r2o_conn[self.num_exc:, :]

        # 脉冲计数相关参数
        self.spike_count_num = spike_count_num
        self.batch_size = batch_size
        self.window_size = window_size
        self.spike_counts = bst.State(jnp.zeros((self.spike_count_num, self.batch_size, self.num_rec)))
        self.temp_spike = bst.State(jnp.zeros((self.window_size, self.batch_size, self.num_rec)))
        self.temp_i = bst.State(0)
        self.spike_count_i = bst.State(0)

        # 输入层到递归层的连接
        ff_init = bst.init.KaimingNormal(scale=7 * (1 - (u.math.exp(-bst.environ.get_dt(), unit_to_scale=u.ms))),
                                       unit=u.mA)
        self.i2r = bst.nn.Sequential(
            bst.nn.Linear(
                num_in, num_rec,
                w_init=ff_init,
                b_init=bst.init.ZeroInit(unit=u.mA),
                w_mask=self.i2r_prob
            ),
            bst.nn.Expon(num_rec, tau=tau_syn, g_initializer=bst.init.Constant(0. * u.mA))
        )

        # 递归层LIF神经元
        self.r = bst.nn.LIF(
            num_rec,
            tau=tau_neu,
            V_reset=0 * u.mV,
            V_rest=0 * u.mV,
            V_th=1. * u.mV,
            spk_fun=bst.surrogate.ReluGrad()
        )

        # 递归层内部兴奋性连接
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

        # 递归层内部抑制性连接
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

        # 输出层
        self.o = bst.nn.Expon(
            num_out, tau=tau_out,
            g_initializer=bst.init.Constant(0.)
        )

        # 递归层到输出层的连接
        self.exc2o = bst.nn.Linear(
            self.num_exc, self.num_out,
            w_init=bst.init.KaimingNormal(scale=1.0),
            w_mask=self.exc2o_conn
        )

    def update(self, spike, ext_current):
        """
        网络前向传播更新
        参数:
            spike: 输入脉冲
            ext_current: 外部电流
        返回:
            输出层活动和递归层膜电位
        """
        rec_spikes = self.r.get_spike()
        e_sps, i_sps = jnp.split(rec_spikes, [self.num_exc], axis=-1)

        self.update_spike_count(rec_spikes)

        # 计算各层电流
        i2r_current = self.i2r(spike)
        exc2r_current = self.exc2r_coba.update(self.exc2r(e_sps), self.r.V.value)
        inh2r_current = self.inh2r_coba.update(self.inh2r(i_sps), self.r.V.value)

        # 合并电流并更新递归层
        r_current = i2r_current + exc2r_current + inh2r_current
        r_current = r_current.at[:, :self.num_exc].set(r_current[:, :self.num_exc] + ext_current)
        self.r(r_current)

        # 更新输出层
        o_current = self.exc2o(e_sps)
        return self.o(o_current), self.r.V.value.mantissa

    def get_weight_matrix(self):
        """获取权重矩阵"""
        exc2r_weight = self.exc2r.layers[0].weight.value['weight'].mantissa
        inh2r_weight = self.inh2r.layers[0].weight.value['weight'].mantissa
        return jnp.concatenate([exc2r_weight, inh2r_weight], axis=0)

    def set_weight_matrix(self, r2r_weight):
        """设置权重矩阵"""
        exc2r_weight = r2r_weight[:self.num_exc]
        inh2r_weight = r2r_weight[self.num_exc:]
        self.exc2r.layers[0].weight.value['weight'] = exc2r_weight * u.mS
        self.inh2r.layers[0].weight.value['weight'] = inh2r_weight * u.mS

    def start_spike_count(self):
        """初始化脉冲计数"""
        self.spike_counts.value = jnp.zeros((self.spike_count_num, self.batch_size, self.num_rec))
        self.temp_spike.value = jnp.zeros((self.window_size, self.batch_size, self.num_rec))
        self.temp_i.value = 0
        self.spike_count_i.value = 0

    def update_spike_count(self, spike):
        """更新脉冲计数"""
        self.temp_spike.value = self.temp_spike.value.at[self.temp_i.value].set(spike)
        self.temp_i.value = self.temp_i.value + 1
        bst.compile.cond(self.temp_i.value == self.window_size - 1, 
                        self.update_spike_count_i, lambda: None)

    def update_spike_count_i(self):
        """更新脉冲计数索引"""
        self.temp_i.value = 0
        self.spike_counts.value = self.spike_counts.value.at[self.spike_count_i.value].set(
            self.temp_spike.value.sum(axis=0))
        self.spike_count_i.value = self.spike_count_i.value + 1

    def get_spike_counts(self):
        """获取脉冲计数"""
        return self.spike_counts.value
