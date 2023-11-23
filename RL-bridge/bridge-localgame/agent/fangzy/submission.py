# -*- coding:utf-8  -*-
# Time  : 2021/5/31 下午4:14
# Author: Yahui Cui
import numpy as np
"""
# =================================== Important =========================================
Notes:
1. this agent is random agent , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""

def my_controller(observation, action_space, is_act_continuous=False):
    if observation['obs']:
        hands_rep = observation['obs']['observation'][:208]  # 每位玩家的手牌
        trick_rep = observation['obs']['observation'][208:416]  # 当前牌墩
        hidden_cards_rep = observation['obs']['observation'][416:468]  # 不可见的牌
        vul_rep = observation['obs']['observation'][468:472]  # 有局方
        dealer_rep = observation['obs']['observation'][472:476]  # 庄家
        current_player_rep = observation['obs']['observation'][476:480]  # 当前玩家
        is_bidding_rep = observation['obs']['observation'][480]  # 下注回合
        bidding_rep = observation['obs']['observation'][481:521]  # 下注
        last_bid_rep = observation['obs']['observation'][521:560]  # 上一次的下注
        bid_amount_rep = observation['obs']['observation'][560:568]  # 下注的数量
        trump_suit_rep = observation['obs']['observation'][568:]  # 将牌花色

        player_id = np.where(current_player_rep == 1)[0][0]
        self_rep = hands_rep[player_id*52:(player_id+1)*52]
        legal_card = observation['obs']['action_mask'][39:]

        agent_action = [[0 for i in range(91)]]
        if not is_bidding_rep:
            agent_action[0][36] = 1
        else:
            card_action = get_card_action(legal_card, trick_rep, player_id, trump_suit_rep)
            agent_action[0][39 + card_action] = 1
    else:
        agent_action = []
        for i in range(len(action_space)):
            action_ = sample_single_dim(action_space[i], is_act_continuous)
            agent_action.append(action_)

        '''
        Cards:
        C,2~10~A
        D,
        H,
        S
        
        bidding:
        1~5
        1C 1D 1H 1S 1NT
        6~10
        2C 2D 2H 2S 2NT
        ......
        30~35
        7C 7D 7H 7S 7NT
        '''

    return agent_action

def get_card_action(legal_card, trick_rep, player_id, trump_suit_rep):
    '''
        看一下对手出的最高的
        如果没有更高的，或者已经被对家超过，那就出最小的
        否则，出比对面大的牌中最小的
    '''
    avail_cards = np.where(legal_card == 1)[0]
    trick_cards = []
    partner_id = (2 + player_id) % 4
    opponent_ids = [(3 + player_id) % 4, (1 + player_id) % 4]
    for i in range(4):
        tmp_cards = trick_rep[i * 52: i * 52 + 52]
        if np.max(tmp_cards) == 0:
            trick_cards.append(-1)
        else:
            trick_cards.append(np.where(tmp_cards==1)[0][0])

    opponent_value = max([eval_card(trick_cards[opponent_ids[i]], trump_suit_rep) for i in range(2)])
    partner_value = eval_card(trick_cards[partner_id], trump_suit_rep)

    if partner_value > opponent_value:
        '''
        出最小的
        '''
        p = avail_cards[0]
        min_value = eval_card(p, trump_suit_rep)
        for i in range(len(avail_cards)):
            t = eval_card(avail_cards[i], trump_suit_rep)
            if t < min_value:
                p = avail_cards[i]
                min_value = t
        return p
    else:
        p = avail_cards[0]
        p_min = p_max = p_max_min = p
        min_value = eval_card(p, trump_suit_rep)
        max_value = min_value
        max_min_value = 999 if max_value < opponent_value else max_value
        for i in range(len(avail_cards)):
            t = eval_card(avail_cards[i], trump_suit_rep)
            if t < min_value:
                p_min = avail_cards[i]
                min_value = t
            if t > max_value:
                p_max = avail_cards[i]
                max_value = t
            if t > opponent_value and t < max_min_value:
                p_max_min = avail_cards[i]
                max_min_value = t
        if max_value > opponent_value:
            return p_max_min
        else:
            return p_min

def eval_card(card_id, trump_suit_rep):
    if card_id == -1:
        return -1
    value = (card_id % 13 / 13) + (card_id // 13) / 100
    card_color = card_id // 13
    if np.max(trump_suit_rep) == 0:
        trump_color = -1
    else:
        trump_color = np.where(trump_suit_rep==1)[0][0]
    if card_color == trump_color:
        value += 1

    return value

def sample_single_dim(action_space_list_each, is_act_continuous):
    each = []
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
    return each


def sample(action_space_list_each, is_act_continuous):
    player = []
    if is_act_continuous:
        for j in range(len(action_space_list_each)):
            each = action_space_list_each[j].sample()
            player.append(each)
    else:
        player = []
        for j in range(len(action_space_list_each)):
            # each = [0] * action_space_list_each[j]
            # idx = np.random.randint(action_space_list_each[j])
            if action_space_list_each[j].__class__.__name__ == "Discrete":
                each = [0] * action_space_list_each[j].n
                idx = action_space_list_each[j].sample()
                each[idx] = 1
                player.append(each)
            elif action_space_list_each[j].__class__.__name__ == "MultiDiscreteParticle":
                each = []
                nvec = action_space_list_each[j].high
                sample_indexes = action_space_list_each[j].sample()

                for i in range(len(nvec)):
                    dim = nvec[i] + 1
                    new_action = [0] * dim
                    index = sample_indexes[i]
                    new_action[index] = 1
                    each.extend(new_action)
                player.append(each)
    return player