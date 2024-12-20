import brainunit as u
import brainstate as bst
import brainpy as bp
import jax.numpy as jnp


class SNN(bst.nn.DynamicsGroup):
    def __init__(self, num_in, num_rec, num_out):
        # 初始化父类DynamicsGroup
        super(SNN, self).__init__()

        # 参数定义
        self.num_in = num_in  # 输入层神经元数量
        self.num_rec = num_rec  # 递归层神经元数量
        self.num_out = num_out  # 输出层神经元数量
        self.last_rec_spikes = u.math.zeros((num_rec,))

        # 定义从输入层到递归层的连接（突触: i->r）
        # 使用Sequential将线性层和指数衰减层连接在一起
        self.i2r = bst.nn.Sequential(
            # 线性层：用于将输入信号映射到递归层
            bst.nn.Linear(
                num_in, num_rec,  # 从输入层到递归层的连接数
                w_init=bst.init.KaimingNormal(scale=7 * (1 - (u.math.exp(-bst.environ.get_dt(), unit_to_scale=u.ms))),
                                              unit=u.mA),  # 使用Kaiming Normal初始化权重
                b_init=bst.init.ZeroInit(unit=u.mA)  # 偏置初始化为零
            ),
            # 指数衰减层：对信号进行时间上的衰减，使其符合生物神经元动力学
            bst.nn.Expon(num_rec, tau=50. * u.ms, g_initializer=bst.init.Constant(0. * u.mA))
        )

        # 定义递归层（r），采用LIF神经元模型
        self.r = bst.nn.LIF(
            num_rec,  # 递归层神经元数量
            tau=50 * u.ms,  # 时间常数，控制膜电位衰减速率
            V_reset=0 * u.mV,  # 膜电位复位值
            V_rest=0 * u.mV,  # 静息膜电位
            V_th=1. * u.mV,  # 膜电位阈值，超过此值时神经元发放脉冲
            spk_fun=bst.surrogate.ReluGrad()  # 近似求导函数，用于实现脉冲发放
        )

        self.r2r = bst.nn.Sequential(
            bst.nn.Linear(
                num_rec, num_rec,
                w_init=bst.init.KaimingNormal(scale=7 * (1 - (u.math.exp(-bst.environ.get_dt(), unit_to_scale=u.ms))),
                                              unit=u.mA),
                b_init=bst.init.ZeroInit(unit=u.mA)
            ),
            bst.nn.Expon(num_rec, tau=50. * u.ms, g_initializer=bst.init.Constant(0. * u.mA))
        )

        # 定义从递归层到输出层的连接（突触: r->o），采用线性层
        # self.r2o = bst.nn.Sequential(
        #     bst.nn.Linear(
        #         num_rec, num_out,          # 从递归层到输出层的连接数
        #         w_init=bst.init.KaimingNormal(scale=7*(1-(u.math.exp(-bst.environ.get_dt(), unit_to_scale=u.ms))), unit=u.mA)  # 使用Kaiming Normal初始化权重
        #     ),
        #     bst.nn.Expon(num_out, tau=5. * u.ms, g_initializer=bst.init.Constant(0. * u.mA))
        # )

        self.r2o = bst.nn.Linear(
            num_rec, num_out,  # 从递归层到输出层的连接数
            w_init=bst.init.KaimingNormal()  # 使用Kaiming Normal初始化权重
        )

        # 定义输出层（o），使用指数衰减层模拟输出信号的时间衰减
        # self.o = bst.nn.LIF(
        #     num_out,                    # 输出层神经元数量
        #     tau=5. * u.ms,             # 时间常数，控制输出信号的衰减速率
        #     V_reset=0. * u.mV,         # 膜电位复位值
        #     V_rest=0. * u.mV,          # 静息膜电位
        #     V_th=1. * u.mV,            # 膜电位阈值，超过此值时神经元发放脉冲
        #     spk_fun=bst.surrogate.ReluGrad()  # 近似求导函数，用于实现脉冲发放
        # )
        self.o = bst.nn.Expon(
            num_out,  # 输出层神经元数量
            tau=50. * u.ms,  # 时间常数，控制输出信号的衰减速率
            g_initializer=bst.init.Constant(0.)  # 初始化电流为零
        )

    # update方法：用于执行网络的一次更新，返回输出层的输出
    def update(self, spike):
        # 依次通过 i2r、r、r2o 和 o 计算输出
        rec_spike = self.r.get_spike()
        current = self.i2r(spike) + self.r2r(rec_spike)

        return self.o(self.r2o(self.r(current)))

    # predict方法：用于预测并获取递归层的膜电位值、脉冲输出和最终输出
    def predict(self, spike):
        r2r_spike = self.r.get_spike()
        current = self.i2r(spike) + self.r2r(r2r_spike)

        # 计算递归层的脉冲输出
        rec_spikes = self.r(current)

        # 计算最终输出
        out = self.o(self.r2o(rec_spikes))

        # 返回递归层的膜电位值、递归层脉冲输出和最终输出
        return self.r.V.value, rec_spikes, out


class SNN_ext(bst.nn.DynamicsGroup):
    def __init__(self, num_in, num_rec, num_out):
        # 初始化父类DynamicsGroup
        super(SNN_ext, self).__init__()

        # 参数定义
        self.num_in = num_in  # 输入层神经元数量
        self.num_rec = num_rec  # 递归层神经元数量
        self.num_out = num_out  # 输出层神经元数量
        self.last_rec_spikes = u.math.zeros((num_rec,))

        # 定义从输入层到递归层的连接（突触: i->r）
        # 使用Sequential将线性层和指数衰减层连接在一起
        self.i2r = bst.nn.Sequential(
            # 线性层：用于将输入信号映射到递归层
            bst.nn.Linear(
                num_in, num_rec,  # 从输入层到递归层的连接数
                w_init=bst.init.KaimingNormal(scale=7 * (1 - (u.math.exp(-bst.environ.get_dt(), unit_to_scale=u.ms))),
                                              unit=u.mA),  # 使用Kaiming Normal初始化权重
                b_init=bst.init.ZeroInit(unit=u.mA)  # 偏置初始化为零
            ),
            # 指数衰减层：对信号进行时间上的衰减，使其符合生物神经元动力学
            bst.nn.Expon(num_rec, tau=50. * u.ms, g_initializer=bst.init.Constant(0. * u.mA))
        )

        # 定义递归层（r），采用LIF神经元模型
        self.r = bst.nn.LIF(
            num_rec,  # 递归层神经元数量
            tau=50 * u.ms,  # 时间常数，控制膜电位衰减速率
            V_reset=0 * u.mV,  # 膜电位复位值
            V_rest=0 * u.mV,  # 静息膜电位
            V_th=1. * u.mV,  # 膜电位阈值，超过此值时神经元发放脉冲
            spk_fun=bst.surrogate.ReluGrad()  # 近似求导函数，用于实现脉冲发放
        )

        self.r2r = bst.nn.Sequential(
            bst.nn.Linear(
                num_rec, num_rec,
                w_init=bst.init.KaimingNormal(scale=7 * (1 - (u.math.exp(-bst.environ.get_dt(), unit_to_scale=u.ms))),
                                              unit=u.mA),
                b_init=bst.init.ZeroInit(unit=u.mA)
            ),
            bst.nn.Expon(num_rec, tau=50. * u.ms, g_initializer=bst.init.Constant(0. * u.mA))
        )

        # 定义从递归层到输出层的连接（突触: r->o），采用线性层
        # self.r2o = bst.nn.Sequential(
        #     bst.nn.Linear(
        #         num_rec, num_out,          # 从递归层到输出层的连接数
        #         w_init=bst.init.KaimingNormal(scale=7*(1-(u.math.exp(-bst.environ.get_dt(), unit_to_scale=u.ms))), unit=u.mA)  # 使用Kaiming Normal初始化权重
        #     ),
        #     bst.nn.Expon(num_out, tau=5. * u.ms, g_initializer=bst.init.Constant(0. * u.mA))
        # )

        self.r2o = bst.nn.Linear(
            num_rec, num_out,  # 从递归层到输出层的连接数
            w_init=bst.init.KaimingNormal()  # 使用Kaiming Normal初始化权重
        )

        # 定义输出层（o），使用指数衰减层模拟输出信号的时间衰减
        # self.o = bst.nn.LIF(
        #     num_out,                    # 输出层神经元数量
        #     tau=5. * u.ms,             # 时间常数，控制输出信号的衰减速率
        #     V_reset=0. * u.mV,         # 膜电位复位值
        #     V_rest=0. * u.mV,          # 静息膜电位
        #     V_th=1. * u.mV,            # 膜电位阈值，超过此值时神经元发放脉冲
        #     spk_fun=bst.surrogate.ReluGrad()  # 近似求导函数，用于实现脉冲发放
        # )
        self.o = bst.nn.Expon(
            num_out,  # 输出层神经元数量
            tau=50. * u.ms,  # 时间常数，控制输出信号的衰减速率
            g_initializer=bst.init.Constant(0.)  # 初始化电流为零
        )

    # update方法：用于执行网络的一次更新，返回输出层的输出
    def update(self, spike, ext_current):
        # 依次通过 i2r、r、r2o 和 o 计算输出
        rec_spike = self.r.get_spike()
        i2r_current = self.i2r(spike)
        r2r_current = self.r2r(rec_spike)
        current = self.i2r(spike) + self.r2r(rec_spike) + ext_current

        return self.o(self.r2o(self.r(current))), self.r.V.value.mantissa

    # predict方法：用于预测并获取递归层的膜电位值、脉冲输出和最终输出
    def predict(self, spike):
        r2r_spike = self.r.get_spike()
        current = self.i2r(spike) + self.r2r(r2r_spike)

        # 计算递归层的脉冲输出
        rec_spikes = self.r(current)

        # 计算最终输出
        out = self.o(self.r2o(rec_spikes))

        # 返回递归层的膜电位值、递归层脉冲输出和最终输出
        return self.r.V.value, rec_spikes, out



class EI(bst.nn.DynamicsGroup):
    def __init__(
        self, num_in, num_rec, num_out, exc_ratio=0.8,
        tau_neu=10 * u.ms, tau_syn=10 * u.ms, tau_out=10 * u.ms,
        ff_scale=1., rec_scale=1., E_exc=3 * u.mV, E_inh=-3 * u.mV,
        e2r_prob=0.5, i2r_prob=0.5, e2o_prob=0.5, i2o_prob=0.5, seed=42,
    ):
        super().__init__()

        self.num_in = num_in
        self.num_rec = num_rec
        self.num_exc = int(num_rec * exc_ratio)
        self.num_inh = num_rec - self.num_exc
        self.num_out = num_out

        # neurons
        self.pop = bst.nn.LIF(
            num_rec, tau=tau_neu,
            V_reset=0 * u.mV, V_rest=0 * u.mV, V_th=1 * u.mV,
            spk_fun=bst.surrogate.ReluGrad()
        )
        ff_init = bst.init.KaimingNormal(scale=ff_scale, unit=u.mA)

        # feedforward
        self.ff2r = bst.nn.Sequential(
            bst.nn.Linear(
                num_in, self.num_exc,
                w_init=ff_init, b_init=bst.init.ZeroInit(unit=u.mA)
            ),
            bst.nn.Expon(self.num_exc, tau=tau_syn, g_initializer=bst.init.Constant(0. * u.mA))
        )

        # recurrent
        inh2r_conn = bst.event.FixedProb(in_size=self.num_inh, out_size=self.num_rec, prob=i2r_prob,
                                         allow_multi_conn=False, seed=seed, weight=1.62 * u.mS)
        exc2r_conn = bst.event.FixedProb(in_size=self.num_exc, out_size=self.num_rec, prob=e2r_prob,
                                         allow_multi_conn=False, seed=seed, weight=-9. * u.mS)
        inh2o_conn = bst.event.FixedProb(in_size=self.num_inh, out_size=self.num_out, prob=i2o_prob,
                                         allow_multi_conn=False, seed=seed, weight=1.62 * u.mS)
        exc2o_conn = bst.event.FixedProb(in_size=self.num_exc, out_size=self.num_out, prob=e2o_prob,
                                         allow_multi_conn=False, seed=seed, weight=-9. * u.mS)

        self.inh2r = bst.nn.AlignPostProj(
            comm=inh2r_conn,
            syn=bst.nn.Expon.desc(self.num_inh, tau=tau_syn),
            out=bst.nn.COBA.desc(E=E_inh),
            post=self.pop
        )
        self.exc2r = bst.nn.AlignPostProj(
            comm=exc2r_conn,
            syn=bst.nn.Expon.desc(self.num_exc, tau=tau_syn),
            out=bst.nn.COBA.desc(E=E_exc),
            post=self.pop
        )

        # output
        self.o = bst.nn.LIF(
            num_out, tau=tau_neu,
            V_reset=0 * u.mV, V_rest=0 * u.mV, V_th=1 * u.mV,
            spk_fun=bst.surrogate.ReluGrad()
        )

        # recurrent to output
        self.inh2o = bst.nn.AlignPostProj(
            comm=inh2o_conn,
            syn=bst.nn.Expon.desc(num_out, tau=tau_syn),
            out=bst.nn.COBA.desc(E=E_inh),
            post=self.o
        )

        self.exc2o = bst.nn.AlignPostProj(
            comm=exc2o_conn,
            syn=bst.nn.Expon.desc(num_out, tau=tau_syn),
            out=bst.nn.COBA.desc(E=E_exc),
            post=self.o
        )

    def update(self, spk_in, current_in):
        # current_in = u.math.tile(current_in, (1, self.num_rec))
        # self.pop(current_in)
        e_sps, i_sps = jnp.split(self.pop.get_spike(), [self.num_exc], axis=-1)
        self.ff2r(spk_in)
        self.exc2r(e_sps)
        self.inh2r(i_sps)

        self.exc2o(e_sps)
        self.inh2o(i_sps)

        return self.o.get_spike()
