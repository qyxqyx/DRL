"""
Actor-Critic reinforcement learning algorithm
Using:
mxnet 0.11.0
gym 0.8.0
"""


import numpy as np
import mxnet as mx

np.random.seed(1)

###############################  Actor  ####################################


class Actor_Critic(object):
    def __init__(self, actions_num, state_dim,
                 actor_lr, critic_lr,  batch_size, gamma, ctx):

        self.gamma        = gamma
        self.ctx          = ctx
        self.actor_lr     = actor_lr
        self.critic_lr    = critic_lr
        self.batch_size   = batch_size
        self.actions_num   = actions_num
        self.state_dim    = state_dim

        self.actor    = self._actor()
        self.critic   = self._critic()


    def _actor_sym(self):
        state = mx.sym.Variable('state')
        action_one_hot = mx.sym.Variable('action')
        td_error = mx.sym.Variable('td_error')

        fc1 = mx.sym.FullyConnected(data=state, num_hidden=10, name='actor_fc1')
        ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='actor_ac1')

        fc2 = mx.sym.FullyConnected(data=ac1, num_hidden=self.actions_num, name='actor_fc2')

        action_prob      = mx.sym.SoftmaxActivation(data=fc2, name='action_prob')

        log_prob         = mx.sym.log(data=action_prob, name='action_log_prob')
        log_prob_one = mx.sym.dot(log_prob, mx.sym.transpose(action_one_hot) )
        log_prob_one     = mx.sym.sum(data=log_prob_one, axis=0)
        loss             = -(log_prob_one * td_error)
        loss             = mx.sym.MakeLoss(data=loss)
        out              = mx.sym.Group([mx.sym.BlockGrad(action_prob), loss])
        return out


    def _critic_sym(self):
        state  = mx.sym.Variable('state')
        v_next = mx.sym.Variable('v_next')
        reward = mx.sym.Variable('reward')

        fc1 = mx.sym.FullyConnected(data=state, num_hidden=10, name='critic_fc1')
        ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='critic_ac1')

        fc2 = mx.sym.FullyConnected(data=ac1, num_hidden=1, name='critic_fc2')
        v_now = mx.sym.Activation(data=fc2, act_type='relu', name='critic_ac2')
        td_error        = reward + self.gamma * v_next - v_now
        td_error_square = mx.sym.square(td_error)
        loss_td         = mx.sym.MakeLoss(td_error_square)

        out = mx.sym.Group([mx.sym.BlockGrad(v_now), loss_td, mx.sym.BlockGrad(td_error)])
        return out


    def _actor(self):
        actor = mx.module.Module(symbol=self._actor_sym(),
                                  data_names=('state',),
                                  label_names=('action', 'td_error'),
                                  context=self.ctx
                                  )

        actor.bind(data_shapes=[('state', (self.batch_size, self.state_dim))],
                    label_shapes=[('action', (self.batch_size, self.actions_num)),
                                  ('td_error', (self.batch_size, ))],
                    for_training=True)
        actor.init_params(initializer=mx.init.Normal(0.1))
        actor.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': self.actor_lr})

        return actor


    def _critic(self):
        critic = mx.module.Module(symbol=self._critic_sym(),
                                   data_names=('state', ),
                                   label_names=('v_next', 'reward'),
                                   context=self.ctx)

        critic.bind(data_shapes=[('state', (self.batch_size, self.state_dim))],
                     label_shapes=[('v_next', (self.batch_size, 1)), ('reward', (self.batch_size, 1))],
                     for_training=True)
        critic.init_params(initializer=mx.init.Normal(0.1))
        critic.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': self.critic_lr})

        return critic


    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = np.zeros(shape=(self.batch_size, self.actions_num))
        td_error = np.zeros(shape=(self.batch_size, ))
        data = mx.io.DataBatch(data=[mx.nd.array(s,self.ctx)],
                               label=[mx.nd.array(a,self.ctx), mx.nd.array(td_error,self.ctx)])
        self.actor.forward(data_batch=data, is_train=False)
        actions_prob = np.squeeze(self.actor.get_outputs()[0].asnumpy()[0])
        action = np.random.choice(np.arange(actions_prob.shape[0]), p=actions_prob.ravel())
        return action

    def learn(self, state, action, reward, state_next):
        state = state[np.newaxis, :]
        state_next = state_next[np.newaxis, :]
        reward = np.array(reward).reshape(self.batch_size,1)

        temp_v_next = np.zeros(shape=(self.batch_size, 1))

        data1 = mx.io.DataBatch(data=[mx.nd.array(state_next, ctx=self.ctx)],
                               label=[mx.nd.array(temp_v_next, ctx=self.ctx),
                                      mx.nd.array(reward, ctx=self.ctx)])
        self.critic.forward(data1, is_train=False)
        v_next = self.critic.get_outputs()[0]

        #
        data2 = mx.io.DataBatch(data=[mx.nd.array(state, ctx=self.ctx)],
                               label=[mx.nd.array(v_next, ctx=self.ctx), mx.nd.array(reward, ctx=self.ctx)])
        self.critic.forward(data2, is_train=True)
        a, b, td_error = self.critic.get_outputs()
        td_error = td_error.asnumpy()
        td_error = td_error.reshape(self.batch_size, )

        self.critic.backward()
        self.critic.update()

        temp = np.zeros(shape=(self.batch_size, self.actions_num))
        temp[0, action] = 1

        data3 = mx.io.DataBatch(data=[mx.nd.array(state, ctx=self.ctx)],
                                label=[mx.nd.array(temp, ctx=self.ctx), mx.nd.array(td_error, ctx=self.ctx)])
        self.actor.forward(data_batch=data3, is_train=True)
        self.actor.backward()
        self.actor.update()



