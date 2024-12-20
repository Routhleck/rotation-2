import brainunit as u
import brainstate as bst
import brainpy as bp
import jax
import jax.numpy as jnp


class SNN_ext(bst.nn.DynamicsGroup):
    def __init__(self, num_in, num_rec, num_out, exc_ratio=0.8,
        tau_neu=10 * u.ms, tau_syn=100 * u.ms, tau_out=10 * u.ms,
        ff_scale=1., rec_scale=1., E_exc=1.6 * u.mV, E_inh=-0.3 * u.mV,
        i2r_prob=0.5, r2r_prob=0.5, r2o_prob=0.1, seed=42,):

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

        # 定义从输入层到递归层的连接（突触: i->r）
        # 使用Sequential将线性层和指数衰减层连接在一起
        ff_init = bst.init.KaimingNormal(scale=7 * (1 - (u.math.exp(-bst.environ.get_dt(), unit_to_scale=u.ms))), unit=u.mA)
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
                w_init=bst.init.KaimingNormal(scale=rec_scale, unit=u.mS),
                b_init=bst.init.ZeroInit(unit=u.mS),
                w_mask=self.exc2r_conn
            ),
            bst.nn.Expon(num_rec, tau=tau_syn, g_initializer=bst.init.Constant(0. * u.mS)),
        )
        self.exc2r_coba = bst.nn.COBA(E_exc)

        self.inh2r = bst.nn.Sequential(
            bst.nn.Linear(
                self.num_inh, self.num_rec,
                w_init=bst.init.KaimingNormal(scale=rec_scale, unit=u.mS),
                b_init=bst.init.ZeroInit(unit=u.mS),
                w_mask=self.inh2r_conn
            ),
            bst.nn.Expon(num_rec, tau=tau_syn, g_initializer=bst.init.Constant(0. * u.mS)),
        )
        self.inh2r_coba = bst.nn.COBA(E_inh)

        # output
        self.o = bst.nn.LIF(
            num_out, tau=tau_out,
            V_reset=0 * u.mV, V_rest=0 * u.mV, V_th=1 * u.mV,
            spk_fun=bst.surrogate.ReluGrad()
        )

        # recurrent to output
        self.exc2o = bst.nn.Sequential(
            bst.nn.Linear(
                self.num_exc, self.num_out,
                w_init=bst.init.KaimingNormal(scale=rec_scale, unit=u.mS),
                b_init=bst.init.ZeroInit(unit=u.mS),
                w_mask=self.exc2o_conn
            ),
            bst.nn.Expon(num_out, tau=tau_syn, g_initializer=bst.init.Constant(0. * u.mS)),
        )
        self.exc2o_coba = bst.nn.COBA(E_exc)

    # update方法：用于执行网络的一次更新，返回输出层的输出
    def update(self, spike, ext_current):
        e_sps, i_sps = jnp.split(self.r.get_spike(), [self.num_exc], axis=-1)

        i2r_current = self.i2r(spike)
        exc2r_current = self.exc2r_coba.update(self.exc2r(e_sps), self.r.V.value)
        inh2r_current = self.inh2r_coba.update(self.inh2r(i_sps), self.r.V.value)
        print(f'i2r_current: {i2r_current.mean().mantissa.primal}')
        print(f'exc2r_current: {exc2r_current.mean().mantissa.primal}')
        print(f'inh2r_current: {inh2r_current.mean().mantissa.primal}')


        r_current = i2r_current + exc2r_current + inh2r_current
        r_current = r_current.at[:, :self.num_exc].set(r_current[:, :self.num_exc] + ext_current)
        print(f'r_current: {r_current.mean().mantissa.primal}')
        self.r(r_current)

        o_current = self.exc2o_coba.update(self.exc2o(e_sps), self.o.V.value)
        print(f'o_current: {o_current.mean().mantissa.primal}')
        self.o(o_current)

        return self.o.get_spike()

    # predict方法：用于预测并获取递归层的膜电位值、脉冲输出和最终输出
    # def predict(self, spike):
    #     r2r_spike = self.r.get_spike()
    #     current = self.i2r(spike) + self.r2r(r2r_spike)
    #
    #     # 计算递归层的脉冲输出
    #     rec_spikes = self.r(current)
    #
    #     # 计算最终输出
    #     out = self.o(self.r2o(rec_spikes))
    #
    #     # 返回递归层的膜电位值、递归层脉冲输出和最终输出
    #     return self.r.V.value, rec_spikes, out
