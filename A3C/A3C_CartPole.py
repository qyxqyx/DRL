"""
Asynchronous Advantage Actor Critic (A3C) Reinforcement Learning example.

The Cartpole example.

Using:
mxnet 0.11.0
gym 0.8.0
"""


import threading
import numpy as np
import gym

import matplotlib.pyplot as plt
import mxnet as mx

MAX_GLOBAL_EP = 1000
UPDATE_GLOBAL_ITER = 10
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

thread_lock = threading.Lock()


class A3C_GLOBAL(object):
    def __init__(self, batch_size, state_dim, actions_num, actor_lr, critic_lr, ctx):
        self.batch_size  = batch_size
        self.state_dim   = state_dim
        self.actions_num = actions_num
        self.actor_lr    = actor_lr
        self.critic_lr   = critic_lr
        self.ctx         = ctx

        self.state = mx.sym.Variable('state')

        self.action_prob = self._actor_mod()
        self.v_current   = self._critic_mod()

        self.actor_params, _  = self.action_prob.get_params()
        self.ctitic_params, _ = self.v_current.get_params()


    def _actor_sym(self):
        fc1 = mx.sym.FullyConnected(data=self.state, num_hidden=200, name='actor_fc1')
        ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='actor_ac1')
        fc2 = mx.sym.FullyConnected(data=ac1, num_hidden=self.actions_num, name='actor_fc2')
        action_prob = mx.sym.SoftmaxActivation(data=fc2, name='action_prob')

        return action_prob


    def _critic_sym(self):
        fc1 = mx.sym.FullyConnected(data=self.state, num_hidden=100, name='critic_fc1')
        ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='critic_ac1')
        fc2 = mx.sym.FullyConnected(data=ac1, num_hidden=1, name='state_value')
        return fc2


    def _actor_mod(self):
        actor = mx.module.Module(symbol=self._actor_sym(),
                                 data_names=('state',),
                                 label_names=None,
                                 context=self.ctx
                                 )
        actor.bind(data_shapes=[('state', (self.batch_size, self.state_dim))],
                   label_shapes=None,
                   for_training=True)
        actor.init_params(initializer=mx.init.Normal(0.1))
        actor.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': self.actor_lr})

        return actor


    def _critic_mod(self):
        critic = mx.module.Module(symbol=self._critic_sym(),
                                   data_names=('state', ),
                                   label_names=None,
                                   context=self.ctx)
        critic.bind(data_shapes=[('state', (self.batch_size, self.state_dim))],
                     label_shapes=None,
                     for_training=True)
        critic.init_params(initializer=mx.init.Normal(0.1))
        critic.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': self.critic_lr})

        return critic




class A3C_WORKER(object):
    def __init__(self, global_mod, worker_index, actions_num, state_dim,
                 entropy_weight, actor_lr, critic_lr, batch_size, gamma, ctx):

        self.global_mod  = global_mod
        self.worker_index= worker_index
        self.gamma       = gamma
        self.entropy_weight = entropy_weight
        self.actor_lr    = actor_lr
        self.critic_lr   = critic_lr
        self.state_dim   = state_dim
        self.actions_num = actions_num
        self.batch_size  = batch_size
        self.ctx         = ctx


        self.action_prob = self._actor_mod()
        self.v_current= self._critic_mod()

        self.actor_params, _ = self.action_prob.get_params()
        self.ctitic_params, _ = self.v_current.get_params()

        self.actor_loss = self._actor_loss_mod()
        self.critic_loss = self._critic_loss_mod()



    def _actor_sym(self, worker_index):
        state = mx.sym.Variable('state')

        fc1 = mx.sym.FullyConnected(data=state, num_hidden=200, name='worker'+str(worker_index)+'_actor_fc1')
        ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='worker'+str(worker_index)+'_actor_ac1')
        fc2 = mx.sym.FullyConnected(data=ac1, num_hidden=self.actions_num, name='worker'+str(worker_index)+'_actor_fc2')
        action_prob = mx.sym.SoftmaxActivation(data=fc2, name='worker'+str(worker_index)+'_action_prob')

        return action_prob


    def _critic_sym(self, worker_index):
        state = mx.sym.Variable('state')

        fc1 = mx.sym.FullyConnected(data=state, num_hidden=100, name='worker'+str(worker_index)+'_critic_fc1')
        ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='worker'+str(worker_index)+'_critic_ac1')
        v_current = mx.sym.FullyConnected(data=ac1, num_hidden=1, name='worker'+str(worker_index)+'_state_value')
        return v_current


    def _actor_loss(self):
        action_prob = mx.sym.Variable('action_prob')
        action_one_hot = mx.sym.Variable('action_one_hot')
        td_error = mx.sym.Variable('td_error')

        log_prob = mx.sym.log(data=action_prob)
        log_prob_one = mx.sym.dot(log_prob, mx.sym.transpose(action_one_hot) )
        log_prob_one = mx.sym.sum(data=log_prob_one, axis=0)
        loss = -(log_prob_one * td_error)

        entropy = action_prob * log_prob
        entropy_loss = mx.sym.sum(entropy, axis=1)
        entropy_loss = self.entropy_weight * entropy_loss
        loss = mx.sym.mean(loss - entropy_loss)

        loss = mx.sym.MakeLoss(data=loss)
        return loss


    def _critic_loss(self):
        v_current = mx.sym.Variable('v_current')
        v_next = mx.sym.Variable('v_next')
        reward = mx.sym.Variable('reward')

        td_error = reward + self.gamma * v_next - v_current
        td_error_square = mx.sym.square(td_error)
        td_error_loss = mx.sym.mean(td_error_square)
        loss_td = mx.sym.MakeLoss(td_error_loss)
        out = mx.sym.Group([mx.sym.BlockGrad(td_error), loss_td])
        return out


    def _actor_mod(self):
        actor = mx.module.Module(symbol=self._actor_sym(self.worker_index),
                                 data_names=('state',),
                                 label_names=None,
                                 context=self.ctx
                                 )
        actor.bind(data_shapes=[('state', (self.batch_size, self.state_dim))],
                   label_shapes=None,
                   for_training=True)
        actor.init_params(initializer=mx.init.Normal(0.1))
        actor.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': self.actor_lr})

        return actor


    def _critic_mod(self):
        critic = mx.module.Module(symbol=self._critic_sym(self.worker_index),
                                  data_names=('state', ),
                                  label_names=None,
                                  context=self.ctx)
        critic.bind(data_shapes=[('state', (self.batch_size, self.state_dim))],
                    label_shapes=None,
                    for_training=True)
        critic.init_params(initializer=mx.init.Normal(0.1))
        critic.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': self.critic_lr})

        return critic


    def _actor_loss_mod(self):
        actor_loss  = mx.module.Module(symbol=self._actor_loss(),
                                       data_names=('action_prob', ),
                                       label_names=('action_one_hot', 'td_error'),
                                       context=self.ctx)
        actor_loss.bind(data_shapes=[('action_prob', (self.batch_size, self.actions_num))],
                        label_shapes=[('action_one_hot', (self.batch_size, self.actions_num)),
                                      ('td_error', (self.batch_size, ))],
                        for_training=True,
                        inputs_need_grad=True
                       )
        actor_loss.init_params(initializer=mx.init.Normal(0.1))
        actor_loss.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': self.actor_lr})

        return actor_loss


    def _critic_loss_mod(self):
        critic_loss = mx.module.Module(symbol=self._critic_loss(),
                                       data_names=('v_current', ),
                                       label_names=('v_next', 'reward'),
                                       context=self.ctx)
        critic_loss.bind(data_shapes=[('v_current', (self.batch_size, 1))],
                         label_shapes=[('v_next', (self.batch_size, 1)), ('reward', (self.batch_size, 1))],
                         for_training=True,
                         inputs_need_grad=True)
        critic_loss.init_params(initializer=mx.init.Normal(0.1))
        critic_loss.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': self.critic_lr})

        return critic_loss


    def get_v_next(self, batch_state_next):
        critic_data = mx.io.DataBatch(data=[mx.nd.array(batch_state_next,self.ctx)], label=None)
        self.v_current.forward(data_batch=critic_data, is_train=None)
        batch_v_next = self.v_current.get_outputs()[0].asnumpy()
        return batch_v_next


    def update_global(self, batch_state, batch_v_next, batch_reward, batch_action_one_hot):
        global thread_lock

        actor_data = mx.io.DataBatch(data=[mx.nd.array(batch_state,self.ctx)], label=None)
        self.action_prob.forward(data_batch=actor_data,is_train=False)
        batch_action_prob = self.action_prob.get_outputs()[0]

        critic_data = mx.io.DataBatch(data=[mx.nd.array(batch_state,self.ctx)], label=None)
        self.v_current.forward(data_batch=critic_data, is_train=None)
        batch_v_current = self.v_current.get_outputs()[0]

        critic_loss_data = mx.io.DataBatch(data=[mx.nd.array(batch_v_current,self.ctx)],
                                           label=[mx.nd.array(batch_v_next,self.ctx),
                                                  mx.nd.array(batch_reward, self.ctx)])
        self.critic_loss.forward(data_batch=critic_loss_data, is_train=True)
        batch_td_error = self.critic_loss.get_outputs()[0].asnumpy()
        # print(batch_td_error.shape[0])
        batch_td_error = batch_td_error.reshape(batch_td_error.shape[0], )
        self.critic_loss.backward()
        critic_grad = self.critic_loss.get_input_grads()[0]

        actor_loss_data = mx.io.DataBatch(data=[mx.nd.array(batch_action_prob,self.ctx)],
                                          label=[mx.nd.array(batch_action_one_hot,self.ctx),
                                                 mx.nd.array(batch_td_error, self.ctx)])
        self.actor_loss.forward(data_batch=actor_loss_data, is_train=True)
        actor_loss = self.actor_loss.get_outputs()[0]
        self.actor_loss.backward()
        actor_grad = self.actor_loss.get_input_grads()[0]

        thread_lock.acquire()
        self.global_mod.action_prob.forward(data_batch=actor_data, is_train=True)
        self.global_mod.action_prob.backward([actor_grad])
        self.global_mod.action_prob.update()
        self.global_mod.v_current.forward(data_batch=critic_data, is_train=True)
        self.global_mod.v_current.backward([critic_grad])
        self.global_mod.v_current.update()
        thread_lock.release()

        return batch_td_error, actor_loss.asnumpy()


    def pull_global(self):
        global thread_lock

        thread_lock.acquire()
        actor_global_params, _ = self.global_mod.action_prob.get_params()
        critic_global_params, _ = self.global_mod.v_current.get_params()
        thread_lock.release()

        actor_params = {}
        critic_params = {}

        for key, data in actor_global_params.items():
            actor_params['worker' + str(self.worker_index) + '_' + key]  = data
        for key, data in critic_global_params.items():
            critic_params['worker' + str(self.worker_index) + '_' + key] = data

        self.action_prob.init_params(initializer=None,arg_params=actor_params,force_init=True)
        self.v_current.init_params(initializer=None, arg_params=critic_params, force_init=True)


    def choose_action(self, state):
        state = state[np.newaxis, :]
        action_data = mx.io.DataBatch(data=[mx.nd.array(state,self.ctx)], label=None)
        self.action_prob.forward(data_batch=action_data, is_train=False)
        actions_prob = np.squeeze(self.action_prob.get_outputs()[0].asnumpy())

        action = np.random.choice(np.arange(actions_prob.shape[0]), p=actions_prob.ravel())
        return action



class Worker(object):
    def __init__(self, env_name, worker_name, worker_mod, gamma, max_episode, max_step_per_episode):
        self.env = gym.make(env_name).unwrapped
        self.name = worker_name
        self.AC = worker_mod
        self.gamma = gamma
        self.max_episode = max_episode
        self.max_step_per_episode = max_step_per_episode

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP, thread_lock
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            ep_td_eror = 0
            ep_actor_loss = 0
            while True:
                if self.name == 'Worker_0':
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done: r = -20
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)


                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_next = 0
                    else:
                        state_next = s_[np.newaxis, :]
                        v_next = self.AC.get_v_next(state_next)

                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_next = r + self.gamma * v_next
                        buffer_v_target.append(v_next)
                    buffer_v_target.reverse()


                    buffer_r = np.array(buffer_r)
                    buffer_r = buffer_r[np.newaxis, :]
                    buffer_r = buffer_r.reshape(buffer_r.shape[1],1)

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)

                    action_one_hot = np.zeros(shape=(buffer_a.shape[0], 2))
                    for i in range(buffer_a.shape[0]):
                        action_one_hot[i][buffer_a[i]] = 1

                    td_error, actor_loss  = self.AC.update_global(batch_state=buffer_s,
                                                                  batch_action_one_hot=action_one_hot,
                                                                  batch_reward=buffer_r,
                                                                  batch_v_next=buffer_v_target
                                                                  )


                    ep_td_eror += np.sum(td_error, axis=0)
                    ep_actor_loss += actor_loss

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    thread_lock.acquire()
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    thread_lock.release()
                    break


if __name__ == "__main__":
    env_name       = 'CartPole-v0'
    env            = gym.make(env_name)
    state_dim      = env.observation_space.shape[0]
    actions_num    = env.action_space.n

    batch_size     = UPDATE_GLOBAL_ITER
    entropy_weight = 0.001
    actor_lr       = 0.001
    critic_lr      = 0.001
    gamma          = 0.9
    entropy_weight = 0.001
    ctx            = mx.cpu(0)
    number_workers = 8


    global_mod = A3C_GLOBAL(batch_size, state_dim, actions_num, actor_lr, critic_lr, ctx)

    workers = []
    # Create worker
    for i in range(number_workers):
        worker_name = 'Worker_%i' % i   # worker name
        worker_mod = A3C_WORKER(global_mod, i, actions_num, state_dim,entropy_weight,
                                 actor_lr, critic_lr, batch_size, gamma, ctx)
        workers.append(Worker(env_name, worker_name, worker_mod, gamma, 100, 200))

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)

    for worker_thread in worker_threads:
        worker_thread.join()

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
