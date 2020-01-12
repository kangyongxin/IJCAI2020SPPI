# Note
四个模型分别是啥？如何构建？如何训练？

编码模型
记忆模型
策略模型
解码模型

先在基本功能上包装小的模块，然后把小模块用到更大的组合中去

## 常见名词解析

tf.contrib.framework.nest

absl 在哪儿？干啥用的？ flags app logging 函数的入口启动和日志记录。打辅助的。
    
    flags可以帮助我们通过命令行来动态的更改代码中的参数。

    一个demo（节选）：

    from absl import app, flags, logging
     
    flags.DEFINE_string('type', '','input type.')
    flags.DEFINE_integer('index', 0,'input idnex')
     
    FLAGS = flags.FLAGS
     
    print(FLAGS.type)
    print(FLAGS.index)

    在命令行输入

    python test.py  --index=0 --type=ps

    结果：

    ps
    0

batch_size = FLAGS.batch_size #应该有不止一个环境和智能体同时进行训练，默认16个，那么怎么把他们合起来呢？ 为啥把他们合起来


collection 又是个啥？
    collections是Python内建的一个集合模块，提供了许多有用的集合类。
    >>> from collections import namedtuple
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> p = Point(1, 2)
    >>> p.x
    1
    >>> p.y
    2
    namedtuple是一个函数，它用来创建一个自定义的tuple对象，并且规定了tuple元素的个数，并可以用属性而不是索引来引用tuple的某个元素。

    这样一来，我们用namedtuple可以很方便地定义一种数据类型，它具备tuple的不变性，又可以根据属性来引用，使用十分方便。

    还有Counter，OrderedDict，defaultdict，deque 等


sonnet as snt 是啥：
    Sonnet是基于TensorFlow的一个库，可用于方便地构建复杂的神经网络，git地址为：https://github.com/deepmind/sonnet
    有一些关于mlp cnn lstm 的基础定义

agent 构造：
    初始化的时候有两个基础函数，ImageEncoderDecoder（这个东西里有关于图像的编码解码部分的网络结构，下面的_encode和_decode 用的到）和_RMACore(step 中用到)
    
    _encode 输入是_prepare_observations（其中包含了，observation，last_reward, last_action三部分）
        encode 里用到了_convnet= snt.nets.ConvNet2D（）输出是16*32只是用来编码observation的
        action 是用onehot 编码得到的
        reward是用了一个叫expand_dim 的函数得到的
        features就是把他们三个concat起来

    _decode 也有三部分：
        decode 用来解码图像部分，action reward都是线性解码
        重构的观测recons包括三个部分：image=image_recon，last_reward=reward_recon,last_action=action_recon

    step 输入是_encode 得到的编码feature，


_RMACore 构建，**这部分是我们要用图解钩代替的地**

    _memory_reader/writer:memory.py 
    
    用到了snt.RNNCore（output, next_state = self._build(input, prev_state) ） 来构造网络

    读取部分用到了一个k紧邻的索引，最好奇的是怎么找到相关的记忆的，read_from_memory()基于余弦相似性内容的记忆矩阵读取，谁和谁的余弦相似
        memory 和 read query 两部分，都进行归一化，然后矩阵乘
        之后用read_strengths进行了一次加权
        经过softmax 之后成为 read weights
        输入四个：
            read_keys, read_strengths, mem_state, top_k)
        输出四个：
            memory_reads, read_weights, read_indices, read_strengths
    read_keys 是 用 read inputs 算出来的，怎么算还得细看
        memory_word_size 200
        num_read_heads 3
        output_dim 603

        read_inputs (16, 512)
        mem_state (16, 1000, 200)
        flat_outputs (16, 603)
        h 3
        flat_keys Tensor("rma_agent/loss/rnn/rma_core_22/memory_reader/strided_slice:0", shape=(16, 600), dtype=float32)
        read_strengths (16, 3)
        read_shape (3, 200)
        read_keys (16, 3, 200)

        _top_k 50
        memory_reads.shape (16, 3, 200)
        read_weights (16, 3, 50)
        read_indices (16, 3, 50)
        read_strength (16, 3)
        这里的read inputs 有512维， 要与mem_state中的

rma.py 是结构
main.py 是流程


# 总结

## 整体思路

1. 用当前策略和记忆产生新的数据： 轨迹的生成以agent.step 为主产生动作，在env.stepa(action)中执行动作，之后记录【s，a，r，baseline】；
2. 用产生的数据构造loss 训练各个网络： 用baseline构造 tvt reward, 中和tvt_reward 和 数据记录中的reward，构造各种loss 训练网络。

## 1.执行阶段

step_outputs, state = agent.step(reward_ph, observation_ph, state_ph)

### 1.1 编码

_, features = self._encode(observation, reward, prev_state.prev_action)

feature 由 观测，奖励，动作 组成

### 1.2 计算

core_outputs, next_core_state = self._core(features, prev_state.core_state)

其中：self._core = _RMACore： 是个RNN结构 。 并且完成对外部存储的读写

core_outputs = [ z, read_info, poliy, action, baseline]
h_next = ['controller_outputs', 'h_controller']+['memory', 'mem_reads', 'h_mem_writer']

1.2.1 
构造的状态z 包含三个部分： 特征features，上一个隐含变量h_prev.controller_outputs，上一次记忆h_prev.mem_reads。并进行了一次mlp编码。

1.2.2
用z 和 h_prev.h_controller 经过RNN来产生下一个 controller_out, h_controller

1.2.3
由controller out 到 h_prev.memory 中去检索，得到 mem_reads, 和 read info.

把controller out 和 mem_read 组合到一起作为 policy_extra_input

1.2.4
策略网络self._policy 根据 当前状态z, 和 policy_extra_input得到  policy_outputs

策略网络的输出包含三个部分：{策略policy，动作action， baseline}

其中　
policy = MLP(tf.concat([shared_inputs, extra_policy_inputs])
action = tf.multinomial(policy,
baseline 的计算： 用z（shareinputs）+ policy作为输入 算出来的。


## 2.训练

### 2.1 tvt reward 计算

输入：read info, baselines


### 2.2 loss 构造和回传
有三个　loss, update_op, loss_logs


total_loss, agent_loss_log　＝　loss(self, observations, rewards, actions,additional_rewards=None)

total_loss = a2c_loss + recon_loss + read_reg_loss

 其中　
 a2c_loss, a2c_loss_extra = trfl.sequence_advantage_actor_critic_loss

 recon_loss, recon_logged_values = losses.reconstruction_losses　＃　recon_loss 由　image_loss + action_loss + reward_loss三部分组成

read_reg_loss = strength_loss + key_norm_loss　应该是一些正则　Computes the sum of read strength and read key regularization lossesComputes the sum of read strength and read key regularization losses



finish 











