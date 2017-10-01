"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""


import numpy as np
import mxnet as mx

np.random.seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 70
MAX_EP_STEPS = 400
LR_A = 0.01  # learning rate for actor
LR_C = 0.01  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300


###############################  Actor  ####################################


class DDPG(object):
    def __init__(self, action_dim, state_dim, action_bound, actor_hidden_num,
                 critic_hidden_num_V,critic_hidden_num_A,beta,
                 actor_lr, critic_lr, memory_size, batch_size, gamma, ctx):
        self.memory = np.zeros((memory_size, state_dim * 2 + action_dim + 1), dtype=np.float32)
        self.memory_size = memory_size
        self.pointer = 0
        self.actor_replace_counter  =0
        self.critic_replace_counter = 0

        assert type(actor_hidden_num) == list
        assert type(critic_hidden_num_V) == list
        assert type(critic_hidden_num_A) == list
        self.actor_hidden_num  = actor_hidden_num
        self.critic_hidden_num_V = critic_hidden_num_V
        self.critic_hidden_num_A = critic_hidden_num_A

        self.beta         = beta
        self.gamma        = gamma
        self.ctx          = ctx
        self.actor_lr     = actor_lr
        self.critic_lr    = critic_lr
        self.batch_size   = batch_size
        self.action_dim   = action_dim
        self.state_dim    = state_dim
        self.action_bound = action_bound

        self.State        = mx.sym.Variable('state')
        self.State_next   = mx.sym.Variable('state_next')
        self.Reward       = mx.sym.Variable('reward')

        self.actor_eval    = self._actor_eval()
        self.actor_target  = self._actor_target()
        self.critic_eval   = self._critic_eval()
        self.critic_target = self._critic_target()

    def _actor_sym(self):
        state = mx.sym.Variable('state')

        for i,num in enumerate(self.actor_hidden_num):
            if i == 0:
                fc1 = mx.sym.FullyConnected(data=state, num_hidden=num, name='actor_fc1')
                ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='actor_ac1')
            else :
                fc1 = mx.sym.FullyConnected(data=ac1, num_hidden=num, name='actor_fc'+str(i+1))
                ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='actor_ac'+str(i+1))

        fc2 = mx.sym.FullyConnected(data=ac1, num_hidden=self.action_dim, name='actor_fc_out')
        ac2 = mx.sym.Activation(data=fc2, act_type='tanh', name='actor_ac_out')
        return ac2

    def _critic_sym(self, for_train=False):
        state = mx.sym.Variable('state')
        action = mx.sym.Variable('action')

        for i,num in enumerate(self.critic_hidden_num_V):
            if i == 0:
                fc1_V = mx.sym.FullyConnected(data=state, num_hidden=num, name='critic_fc1_V')
                ac1_V = mx.sym.Activation(data=fc1_V, act_type='relu', name='critic_ac1_V')
            else :
                fc1_V = mx.sym.FullyConnected(data=ac1_V, num_hidden=num, name='critic_fc'+str(i+1)+'_V')
                ac1_V = mx.sym.Activation(data=fc1_V, act_type='relu', name='actor_ac'+str(i+1)+'_V')

        for i,num in enumerate(self.critic_hidden_num_A):
            if i == 0:
                fc1_A = mx.sym.FullyConnected(data=action, num_hidden=num, name='critic_fc1_A')
                ac1_A = mx.sym.Activation(data=fc1_A, act_type='relu', name='critic_ac1_A')
            else :
                fc1_A = mx.sym.FullyConnected(data=ac1_A, num_hidden=num, name='critic_fc'+str(i+1)+'_A')
                ac1_A = mx.sym.Activation(data=fc1_A, act_type='relu', name='actor_ac'+str(i+1)+'_A')

        fc1 = ac1_V + ac1_A
        fc2 = mx.sym.FullyConnected(data=fc1, num_hidden=1, name='critic_fc_out')
        q_value = mx.sym.Activation(data=fc2, act_type='tanh', name='critic_ac_out')

        if for_train == True:
            reward = mx.sym.Variable('reward')
            q_next = mx.sym.Variable('q_next')

            td_error = reward + self.gamma * q_next - q_value
            td_error_square = mx.sym.square(td_error)
            loss_td = mx.sym.MakeLoss(td_error_square)
            # loss_q  = mx.sym.MakeLoss(-q_value)
            out = mx.sym.Group([mx.sym.BlockGrad(q_value), loss_td, mx.sym.BlockGrad(-td_error)])
            return out
        else:
            return q_value


    def _actor_eval(self):
        actor_eval = mx.module.Module(symbol=self._actor_sym(),
                                      data_names=('state',),
                                      label_names=None,
                                      context=self.ctx
                                      )

        actor_eval.bind(data_shapes=[('state', (self.batch_size, self.state_dim))],
                        label_shapes=None,
                        for_training=True)
        actor_eval.init_params(initializer=mx.init.Normal(0.3))
        actor_eval.init_optimizer(optimizer='RMSProp', optimizer_params={'learning_rate': self.actor_lr})

        return actor_eval

    def _actor_target(self):
        actor_target = mx.module.Module(symbol=self._actor_sym(),
                                        data_names=['state',],
                                        label_names=None,
                                       context=self.ctx)

        actor_target.bind(data_shapes=[('state', (self.batch_size, self.state_dim))],
                          label_shapes=None,
                          for_training=False)
        actor_target.init_params(initializer=mx.init.Normal(0.3))
        actor_target.init_optimizer(optimizer='RMSProp', optimizer_params={'learning_rate': self.actor_lr})

        return actor_target

    def _critic_eval(self):
        critic_eval = mx.module.Module(symbol=self._critic_sym(for_train=True),
                                       data_names=('state', 'action'),
                                       label_names=('reward', 'q_next'),
                                       context=self.ctx)

        critic_eval.bind(data_shapes=[('state', (self.batch_size, self.state_dim)),('action', (self.batch_size, self.action_dim))],
                         label_shapes=[('reward', (self.batch_size, 1)),('q_next', (self.batch_size, 1))],
                         for_training=True,
                         inputs_need_grad=True)
        critic_eval.init_params(initializer=mx.init.Normal(0.3))
        critic_eval.init_optimizer(optimizer='RMSProp', optimizer_params={'learning_rate': self.critic_lr})

        return critic_eval

    def _critic_target(self):
        critic_target = mx.module.Module(symbol=self._critic_sym(),
                                         data_names=('state', 'action'),
                                         label_names=None,
                                         context=self.ctx)
        critic_target.bind(data_shapes=[('state', (self.batch_size, self.state_dim)),('action', (self.batch_size, self.action_dim))],
                           label_shapes=None,
                           for_training=False)
        critic_target.init_params(initializer=mx.init.Normal(0.3))
        critic_target.init_optimizer(optimizer='RMSProp', optimizer_params={'learning_rate': self.critic_lr})

        return critic_target

    def choose_action(self, s):
        s = s[np.newaxis, :]
        self.actor_eval.forward(mx.io.DataBatch([mx.nd.array(s,self.ctx)]), is_train=False)
        actions = self.actor_eval.get_outputs()[0].asnumpy()[0]
        return actions

    def learn(self):
        # hard replace parameters
        new_params, _ = self.actor_eval.get_params()
        old_params, _ = self.actor_target.get_params()
        #print(new_params)
        for key in new_params:
            old_params[key] = old_params[key] * self.beta + new_params[key] * (1-self.beta)

        self.actor_target.init_params(initializer=None, arg_params=old_params, force_init=True)

        new_params, _ = self.critic_eval.get_params()
        old_params, _ = self.critic_target.get_params()
        #print(new_params)
        for key in new_params:
            old_params[key] = old_params[key] * self.beta + new_params[key] * (1 - self.beta)

        self.critic_target.init_params(initializer=None, arg_params=old_params, force_init=True)


        index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_sample = self.memory[index, :]
        batch_state  = batch_sample[:, :self.state_dim]
        batch_action = batch_sample[:, self.state_dim: self.state_dim + self.action_dim]
        batch_reward = batch_sample[:, -self.state_dim - 1: -self.state_dim]
        batch_state_next = batch_sample[:, -self.state_dim:]

        # 将s_next输入actor_target网络，得出a_next
        # state_date = mx.io.DataBatch([mx.nd.array(batch_state, ctx=self.ctx)])
        state_next_date = mx.io.DataBatch([mx.nd.array(batch_state_next, ctx=self.ctx)])
        self.actor_target.forward(state_next_date,is_train=False)
        action_next = self.actor_target.get_outputs()[0]

        # 将s_next与a_next一起输入critic_target网络，得出q(s_next,a_next)
        data1 = mx.io.DataBatch(data=[mx.nd.array(batch_state_next, ctx=self.ctx), mx.nd.array(action_next, ctx=self.ctx)])
        self.critic_target.forward(data1, is_train=False)
        q_next = self.critic_target.get_outputs()[0]

        #
        data2 = mx.io.DataBatch(data=[mx.nd.array(batch_state, ctx=self.ctx), mx.nd.array(batch_action, ctx=self.ctx)],
                               label=[mx.nd.array(batch_reward, ctx=self.ctx), mx.nd.array(q_next, ctx=self.ctx)])
        self.critic_eval.forward(data2, is_train=True)
        a,b,c= self.critic_eval.get_outputs()

        self.critic_eval.backward()
        self.critic_eval.update()
        diff = self.critic_eval.get_input_grads()[1]
        diff = diff/(2 * mx.nd.array(c, self.ctx))
        diff = [-diff]

        self.actor_eval.backward(diff)
        self.actor_eval.update()

        return a, b


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1



