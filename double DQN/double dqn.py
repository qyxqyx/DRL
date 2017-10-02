"""
Double DQN

Using:
mxnet 0.11.0
"""

import numpy as np
import mxnet as mx

np.random.seed(1)

class double_DQN(object):
    def __init__(self, actions_num, state_dim, net_hidden_num,
                 beta,epislon,
                 lr, memory_size, batch_size, gamma, ctx):
        self.memory = np.zeros((memory_size, state_dim * 2 + 1 + 1), dtype=np.float32)
        self.memory_size = memory_size
        self.pointer = 0

        assert type(net_hidden_num) == list

        self.net_hidden_num  = net_hidden_num

        self.beta         = beta
        self.gamma        = gamma
        self.ctx          = ctx
        self.lr           = lr
        self.batch_size   = batch_size
        self.actions_num  = actions_num
        self.state_dim    = state_dim
        self.epsilon      = epislon
        self.eval         = self._eval()
        self.target       = self._target()
        params, _ = self.eval.get_params()
        self.target.init_params(initializer=None,arg_params=params, force_init=False)

    def _sym(self, for_train=False):
        state = mx.sym.Variable('state')

        for i,num in enumerate(self.net_hidden_num):
            if i == 0:
                fc1 = mx.sym.FullyConnected(data=state, num_hidden=num, name='fc1')
                ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='ac1')
            else :
                fc1 = mx.sym.FullyConnected(data=ac1, num_hidden=num, name='fc'+str(i+1))
                ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='ac'+str(i+1))

        # 输入state，输出当前state下可选择的所有action的q值（Q(s, a_all)）.
        q_value = mx.sym.FullyConnected(data=ac1, num_hidden=self.actions_num, name='q_value')


        if for_train == True:
            # action是一个one——hot的array，共有batch_size行，每一行表示实际选择的action的序号
            action = mx.sym.Variable('action')
            # batch_size个q_target，用来作为Q(s, a)的label
            q_target = mx.sym.Variable('q_target')

            q_predict = q_value * action
            q_predict = mx.sym.sum(data=q_predict, axis=1)
            td_error = q_target - q_predict
            td_error_square = mx.sym.square(td_error)
            loss_td = mx.sym.MakeLoss(td_error_square)

            out = mx.sym.Group([mx.sym.BlockGrad(q_value), loss_td])
            return out
        else:
            return q_value


    def _eval(self):
        eval = mx.module.Module(symbol=self._sym(for_train=True),
                               data_names=('state', ),
                               label_names=('action', 'q_target'),
                               context=self.ctx)

        eval.bind(data_shapes=[('state', (self.batch_size, self.state_dim))],
                 label_shapes=[('action', (self.batch_size, self.actions_num)),('q_target', (self.batch_size, ))],
                 for_training=True
                 )
        eval.init_params(initializer=mx.init.Normal(0.3))
        eval.init_optimizer(optimizer='RMSProp', optimizer_params={'learning_rate': self.lr})

        return eval

    def _target(self):
        target = mx.module.Module(symbol=self._sym(),
                                 data_names=('state',),
                                 label_names=None,
                                 context=self.ctx)
        target.bind(data_shapes=[('state', (self.batch_size, self.state_dim))],
                   label_shapes=None,
                   for_training=False)
        target.init_params(initializer=mx.init.Normal(0.3))
        target.init_optimizer(optimizer='RMSProp', optimizer_params={'learning_rate': self.lr})

        return target

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = np.array(np.zeros(shape=(1, self.actions_num)))
        q_target = np.array([1])

        self.eval.forward(mx.io.DataBatch(data=[mx.nd.array(s,self.ctx)],
                                          label=[mx.nd.array(a,self.ctx), mx.nd.array(q_target,self.ctx)]), is_train=False)
        actions = self.eval.get_outputs()[0].asnumpy()[0]
        action_index = np.argmax(actions)

        if np.random.uniform() > self.epsilon:
            action_index = np.random.randint(0, self.actions_num)
        return action_index


    def learn(self):
        # soft replace parameters
        new_params, _ = self.eval.get_params()
        old_params, _ = self.target.get_params()

        for key in new_params:
            old_params[key] = old_params[key] * self.beta + new_params[key] * (1-self.beta)

        self.target.init_params(initializer=None, arg_params=old_params, force_init=True)

        # 从memory中随机抽取batch_size个(s, r, a, s_)
        index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_sample = self.memory[index, :]

        batch_state  = batch_sample[:, :self.state_dim]
        batch_action = batch_sample[:, self.state_dim: self.state_dim + 1]
        batch_reward = batch_sample[:, -self.state_dim - 1: -self.state_dim]
        batch_state_next = batch_sample[:, -self.state_dim:]

        # 首先得到Q_predict(s_next, a_all), 求出使Q_predict(s_next, a)最大的a
        a = np.array(np.zeros(shape=(self.batch_size, self.actions_num)))
        q_target_temp = np.array(np.zeros(shape=(self.batch_size,)))
        state_next_date = mx.io.DataBatch(data=[mx.nd.array(batch_state_next,self.ctx)],
                                          label=[mx.nd.array(a,self.ctx), mx.nd.array(q_target_temp,self.ctx)])
        self.eval.forward(state_next_date,is_train=False)
        q_next_eval = self.eval.get_outputs()[0].asnumpy()
        action_next = np.argmax(q_next_eval, axis=1)

        # 首先得到Q_target(s_next, a_all)
        data1 = mx.io.DataBatch(data=[mx.nd.array(batch_state_next, ctx=self.ctx)])
        self.target.forward(data1, is_train=False)
        q_next_target = self.target.get_outputs()[0].asnumpy()

        # 根据使用eval网络求出的a，得出q_target
        q_target = np.array(np.zeros(shape=(32, 1)))
        (m, n) = q_next_target.shape
        for i in range(m):
            q_target[i] = q_next_target[i][action_next[i]]
        q_target = self.gamma*q_target + batch_reward
        q_target = q_target.reshape(self.batch_size, )


        # 将action、state、q_target输入eval网络，训练eval网络
        temp = np.zeros(shape=(self.batch_size, self.actions_num))
        for i, j in enumerate(batch_action):
            j = int(j)
            temp[i, j] = 1
        data2 = mx.io.DataBatch(data=[mx.nd.array(batch_state, ctx=self.ctx)],
                               label=[mx.nd.array(temp, ctx=self.ctx), mx.nd.array(q_target, ctx=self.ctx)])
        self.eval.forward(data2, is_train=True)
        a,b= self.eval.get_outputs()
        self.eval.backward()
        self.eval.update()

        return a, b


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1



