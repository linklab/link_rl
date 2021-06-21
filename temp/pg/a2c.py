import numpy as np
import torch


def calc_nstep_returns(rewards, dones, next_v_pred, gamma, n):
    returns = torch.zeros_like(rewards)
    future_return = next_v_pred
    not_dones = 1 - dones
    for t in reversed(range(n)):
        future_return = rewards[t] + gamma * future_return * not_dones[t]
        returns[t] = future_return
    return returns


class ActorCritic:
    def __init__(self):
        self.gamma = 0.99
        self.num_step_returns = 2

    def calc_nstep_advs_v_targets(self, batch, v_preds):
        next_states = batch['next_states'][-1]

        with torch.no_grad():
            next_v_pred = self.calc_v(next_states, use_cache=False)

        v_preds = v_preds.detach() # adv does not accumulate grad

        nstep_returns = calc_nstep_returns(
            batch['rewards'], batch['dones'], next_v_pred, self.gamma, self.num_step_returns
        )
        advs = nstep_returns - v_preds
        v_targets = nstep_returns

        return advs, v_targets

    def calc_policy_loss(self, batch, pdparams, advs):
        return super().calc_policy_loss(batch, pdparams, advs)

    def calc_val_loss(self, v_preds, v_targets):
        assert v_preds.shape == v_targets.shape, f'{v_preds.shape} != {v_targets.shape}'
        val_loss = self.val_loss_coef * self.net.loss_fn(v_preds, v_targets)
        return val_loss

    def train(self):
        clock = self.body.env.clock
        if self.to_train == 1:
            batch = self.sample()
            clock.set_batch_size(len(batch))
            pdparams, v_preds = self.calc_pdparam_v(batch)
            advs, v_targets = self.calc_advs_v_targets(batch, v_preds)
            policy_loss = self.calc_policy_loss(batch, pdparams, advs)
            val_loss = self.calc_val_loss(v_preds, v_targets) # from critic
            if self.shared: # shared network
                loss = policy_loss + val_loss
                self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
            else:
                self.net.train_step(policy_loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
                self.critic_net.train_step(val_loss, self.critic_optim, self.critic_lr_scheduler, clock=clock, global_net=self.global_critic_net)
                loss = policy_loss + val_loss
            # reset
            self.to_train = 0
            return loss.item()
        else:
            return np.nan